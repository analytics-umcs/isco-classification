import base64
import csv
import hmac
import io
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ollama

# Resolve bundled assets and embeddings from the app directory regardless of
# where Streamlit is launched from.
APP_DIR = Path(__file__).resolve().parent

# ============================================================
# KONFIGURACJA STRONY
# ============================================================
def _translate_file_uploader_ui() -> None:
    """Podmienia domyślne (angielskie) napisy widgetu st.file_uploader na
    polskie - Streamlit nie oferuje natywnej lokalizacji tego widgetu.
    Robione przez JS (ten sam mechanizm co _ctrl_enter_shortcut niżej -
    window.parent.document). Przeszukuje CAŁY dokument (a nie jeden konkretny
    data-testid) - selektory Streamlita zmieniają się między wersjami, więc
    poleganie na jednym z nich okazało się zawodne. Dodatkowo próbuje
    kilkukrotnie w krótkich odstępach (setInterval) na wypadek, gdyby widget
    wyrenderował się dopiero chwilę po tym wywołaniu, a MutationObserver łapie
    uploadery pojawiające się później (np. po przełączeniu trybu
    Respondent/Partner albo zmianie strony)."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const exactTranslations = {
            'Drag and drop file here': 'Wczytaj plik',
            'Browse files': 'Przeglądaj pliki',
        };
        function translateNode(node) {
            if (node.nodeType === Node.TEXT_NODE) {
                const original = node.textContent;
                const trimmed = original.trim();
                if (!trimmed) return;
                if (exactTranslations[trimmed]) {
                    node.textContent = original.replace(trimmed, exactTranslations[trimmed]);
                    return;
                }
                const limitMatch = trimmed.match(/^Limit (\\S+) per file(.*)$/);
                if (limitMatch) {
                    node.textContent = original.replace(trimmed, `Limit ${limitMatch[1]} na plik${limitMatch[2]}`);
                }
            } else if (node.childNodes && node.childNodes.length) {
                node.childNodes.forEach(translateNode);
            }
        }
        function translateAll() {
            translateNode(doc.body);
        }
        translateAll();
        if (window.parent.__iscoUploaderInterval) {
            clearInterval(window.parent.__iscoUploaderInterval);
        }
        let ticks = 0;
        window.parent.__iscoUploaderInterval = setInterval(() => {
            translateAll();
            ticks += 1;
            if (ticks > 30) {
                clearInterval(window.parent.__iscoUploaderInterval);
            }
        }, 250);
        if (!window.parent.__iscoUploaderObserver) {
            window.parent.__iscoUploaderObserver = new MutationObserver(translateAll);
            window.parent.__iscoUploaderObserver.observe(doc.body, {childList: true, subtree: true});
        }
        </script>
        """,
        height=0,
    )


st.set_page_config(
    page_title="Klasyfikacja zawodów ISCO-08",
    page_icon="🧭",
    layout="wide",  # zmienione z "centered" - więcej miejsca poziomego, żeby długie nazwy
    # zawodów w liście kandydatów rzadziej wymagały skracania (patrz MAX_LABEL_LINE_LEN)
)
_translate_file_uploader_ui()

EMB_DIR = APP_DIR / "isco_embeddings" / "level_4"  # pełna baza 436 kodów (poziom 4) - używana w modułach 1 i 2
MODEL_NAME = "qwen3-embedding:8b"  # lokalnie przez Ollama, brak API
OLLAMA_BATCH_SIZE = 32

# Qwen3-Embedding wymaga prefiksu instrukcji TYLKO po stronie zapytania (query),
# NIE po stronie dokumentów (korpus ISCO embedowany jest bez prefiksu w
# build_embeddings.py). Instrukcja po angielsku - tak zaleca zespół Qwen dla
# najlepszej jakości nawet przy zapytaniach w innych językach.
QUERY_INSTRUCTION = (
    "Instruct: Given a description of a person's job or duties, retrieve the "
    "matching ISCO-08 occupation group description.\nQuery: "
)

WEIGHTS = {"title": 0.30, "tasks": 0.50, "synteza": 0.20}

# Foldery embeddingów dla trybu kaskadowego (kodowanie cyfra po cyfrze)
LEVEL_EMB_DIRS = {
    1: APP_DIR / "isco_embeddings" / "level_1",   # 10 kodów - grupy główne
    2: APP_DIR / "isco_embeddings" / "level_2",   # 43 kody - grupy drugorzędne
    3: APP_DIR / "isco_embeddings" / "level_3",   # 130 kodów - grupy średnie
    4: APP_DIR / "isco_embeddings" / "level_4",   # 436 kodów - grupy elementarne
}

# Kolumny źródłowe używane do klasyfikacji - osobny zestaw dla respondenta
# i dla jego partnera. Przełącznik "Respondent / Partner" (patrz
# render_mode_selector) decyduje, z którego zestawu korzystają moduły 1 i 2.
TARGET_COLUMNS = {
    "Respondent": {"zawod": "B33", "obowiazki": "B34", "wyksztalcenie": "B35"},
    "Partner": {"zawod": "B48", "obowiazki": "B49", "wyksztalcenie": "B50"},
}

APP_USERS = {
    "User1": "JuPpayQ9",
    "User2": "Qaw8bBIP",
    "User3": "4WjJICEV",
    "User4": "GdpenECK",
    "User5": "rS4xCzrE",
    "User6": "fueIZ8Ab",
    "User7": "BiRPwP3x",
    "User8": "jrkj4Xoh",
    "User9": "qlFhOlqR",
    "User10": "FefEeumO",
}
ADMIN_USERS = {"User1", "User2"}
QUESTIONNAIRE_DB_PATH = Path(
    os.environ.get("QUESTIONNAIRE_DB_PATH", APP_DIR / "data" / "questionnaire.sqlite3")
)
CODING_DB_PATH = Path(
    os.environ.get("CODING_DB_PATH", APP_DIR / "data" / "coding_progress.sqlite3")
)
# Cache roboczych df (per użytkownik) na dysku - pozwala odtworzyć wczytany
# plik i pozycję kodowania po odświeżeniu strony (F5), kiedy st.file_uploader
# i st.session_state tracą swój stan (patrz _persist_df / _load_df_cache niżej).
SESSION_CACHE_DIR = Path(
    os.environ.get("SESSION_CACHE_DIR", APP_DIR / "data" / "session_cache")
)

# Mapowanie klucza stanu sesji (df_state_key) na etykietę wariantu metodologicznego
# używaną w ewidencji postępu kodowania i w pliku przydziału (Status.csv).
# manual_df = Metoda A (kodowanie ręczne), hitl_df = Metoda B (klasyfikacja
# z udziałem eksperta, pełna kaskada), hitl1d_df = Metoda C (1 cyfra przyporządkowana).
DF_STATE_KEY_TO_WARIANT = {
    "manual_df": "Ręczne",
    "hitl_df": "AI",
    "hitl1d_df": "AI (1 cyfra)",
}

WARSAW_TZ = ZoneInfo("Europe/Warsaw")


def _now_pl() -> str:
    """Aktualny czas w polskiej strefie (uwzględnia czas letni/zimowy automatycznie),
    w czytelnym formacie DD.MM.RRRR GG:MM - używane we wszystkich znacznikach czasu
    zapisywanych przez aplikację (ewidencja kodowania, kwestionariusz badawczy)."""
    return datetime.now(WARSAW_TZ).strftime("%d.%m.%Y %H:%M")

ASSET_DIR = APP_DIR / "assets"
LOGO_PATHS = {
    "UMCS": ASSET_DIR / "logoc.png",
    "IFiS PAN": ASSET_DIR / "ifis.jpeg",
    "IFiS PAN Łódź": ASSET_DIR / "lodz.png",
    "ESS": ASSET_DIR / "ess_eric_logo.jpg",
}


# Kolory identyfikacji wizualnej
COLOR_ISCO = "#003B73"  # granat
COLOR_ESS = "#C1121F"   # czerwony

QUESTIONNAIRE_SECTIONS = [
    {
        "id": "A",
        "title": "Stwierdzenia dotyczące codziennych sytuacji",
        "instruction": "Określ, w jakim stopniu zgadzasz się z każdym stwierdzeniem.",
        "options": {
            1: "Zdecydowanie się nie zgadzam", 2: "Nie zgadzam się",
            3: "Raczej się nie zgadzam", 4: "Raczej się zgadzam",
            5: "Zgadzam się", 6: "Zdecydowanie się zgadzam",
        },
        "questions": [
            "Zwykle biorę pod uwagę różne opinie na temat danego zjawiska, nawet wówczas, gdy mam już wyrobiony pogląd.",
            "Unikam niejasnych sytuacji.",
            "Myślę, że dobrze uporządkowane życie jest zgodne z moim temperamentem.",
            "Czuję się źle, kiedy nie rozumiem powodów, dla których pewne sytuacje zdarzają się w moim życiu.",
            "Unikam brania udziału w wydarzeniach, nie wiedząc, czego mogę się po nich spodziewać.",
            "Zwykle podejmuję ważne decyzje szybko i pewnie.",
            "Mógłbym opisać siebie jako osobę niezdecydowaną.",
            "Podejmując większość ważnych decyzji, borykam się z mnóstwem sprzeczności.",
            "Przyglądając się większości sytuacji konfliktowych, potrafię zwykle dostrzec racje obu stron.",
            "Unikam przebywania wśród ludzi, którzy są zdolni do nieoczekiwanych działań.",
            "Dopiero ustalenie spójnych reguł umożliwia mi cieszenie się życiem.",
            "Cenię sobie zorganizowany styl życia.",
            "Czuję dyskomfort, gdy czyjeś czyny lub intencje są dla mnie niejasne.",
            "Zwykle dostrzegam wiele możliwych rozwiązań problemu, przed którym stoję.",
            "Unikam sytuacji, których konsekwencji nie da się przewidzieć.",
        ],
    },
    {
        "id": "B", "title": "Sposoby działania i myślenia",
        "instruction": "Określ, jak często myślisz lub działasz w opisany sposób.",
        "options": {1: "Rzadko / nigdy", 2: "Czasami", 3: "Często", 4: "Prawie zawsze / zawsze"},
        "questions": [
            "Starannie planuję wykonywane zadania.", "Działam bez namysłu.",
            "Trudno mi skupić uwagę.", "Jestem opanowany/a.", "Łatwo się koncentruję.",
            "Jestem rozważny/a.", "Mówię rzeczy bez namysłu.", "Działam pod wpływem chwili.",
        ],
    },
    {
        "id": "C", "title": "Opis siebie",
        "instruction": "Oceń, w jakim stopniu każde określenie odnosi się do Ciebie.",
        "options": {
            1: "Zdecydowanie się nie zgadzam", 2: "Raczej się nie zgadzam",
            3: "W niewielkim stopniu się nie zgadzam", 4: "Ani się zgadzam, ani nie zgadzam",
            5: "W niewielkim stopniu się zgadzam", 6: "Raczej się zgadzam",
            7: "Zdecydowanie się zgadzam",
        },
        "questions": [
            "Postrzegam siebie jako osobę lubiącą towarzystwo innych, aktywną i optymistyczną.",
            "Postrzegam siebie jako osobę krytyczną względem innych, konfliktową.",
            "Postrzegam siebie jako osobę sumienną, zdyscyplinowaną.",
            "Postrzegam siebie jako osobę pełną niepokoju, łatwo wpadającą w przygnębienie.",
            "Postrzegam siebie jako osobę otwartą na nowe doznania, w złożony sposób postrzegającą świat.",
            "Postrzegam siebie jako osobę zamkniętą w sobie, wycofaną i cichą.",
            "Postrzegam siebie jako osobę zgodną, życzliwą.",
            "Postrzegam siebie jako osobę źle zorganizowaną, niedbałą.",
            "Postrzegam siebie jako osobę niemartwiącą się, stabilną emocjonalnie.",
            "Postrzegam siebie jako osobę trzymającą się utartych schematów, biorącą rzeczy wprost.",
        ],
    },
    {
        "id": "D", "title": "Przetwarzanie bodźców i doświadczeń",
        "instruction": "Odpowiedz zgodnie z tym, jak się czujesz (1 — zupełnie nie, 4 — umiarkowanie, 7 — zdecydowanie tak).",
        "options": {1: "Zupełnie nie", 2: "2", 3: "3", 4: "Umiarkowanie", 5: "5", 6: "6", 7: "Zdecydowanie tak"},
        "questions": [
            "Czy masz bogate, złożone życie wewnętrzne?", "Czy drażnią Cię głośne dźwięki?",
            "Czy głęboko przeżywasz sztukę lub muzykę?",
            "Czy denerwujesz się, kiedy musisz zrobić dużo rzeczy jednocześnie?",
            "Czy drażni Cię, kiedy inni chcą od Ciebie zbyt wielu rzeczy naraz?",
            "Czy zmiany w Twoim życiu dezorganizują Cię?",
            "Czy zwracasz uwagę na delikatne lub piękne zapachy, smaki, dźwięki albo dzieła sztuki i cieszysz się nimi?",
            "Czy źle się czujesz, gdy trzeba robić wiele rzeczy jednocześnie?",
            "Czy przeszkadzają Ci intensywne bodźce, np. głośne dźwięki lub chaos?",
            "Czy stajesz się nerwowy/a i niepewny/a, a w efekcie osiągasz gorsze wyniki, gdy ktoś obserwuje Cię podczas rywalizacji lub wykonywania zadania?",
        ],
    },
    {
        "id": "E", "title": "Myśli i odczucia związane ze stresem",
        "instruction": "Wskaż, jak często w ostatnim miesiącu myślałeś/aś lub czułeś/aś się w opisany sposób.",
        "options": {1: "Nigdy", 2: "Prawie nigdy", 3: "Czasem", 4: "Dość często", 5: "Bardzo często"},
        "questions": [
            "Jak często w ciągu ostatniego miesiąca byłeś/aś zdenerwowany/a, ponieważ zdarzyło się coś niespodziewanego?",
            "Jak często w ciągu ostatniego miesiąca czułeś/aś, że ważne sprawy w Twoim życiu wymykają Ci się spod kontroli?",
            "Jak często w ciągu ostatniego miesiąca odczuwałeś/aś zdenerwowanie i napięcie?",
            "Jak często w ciągu ostatniego miesiąca byłeś/aś przekonany/a, że jesteś w stanie poradzić sobie z problemami osobistymi?",
            "Jak często w ciągu ostatniego miesiąca czułeś/aś, że sprawy układają się po Twojej myśli?",
            "Jak często w ciągu ostatniego miesiąca stwierdzałeś/aś, że nie radzisz sobie ze wszystkimi obowiązkami?",
            "Jak często w ciągu ostatniego miesiąca potrafiłeś/aś opanować swoje rozdrażnienie?",
            "Jak często w ciągu ostatniego miesiąca czułeś/aś, że wszystko Ci wychodzi?",
            "Jak często w ciągu ostatniego miesiąca złościłeś/aś się, ponieważ nie miałeś/aś wpływu na to, co się zdarzyło?",
            "Jak często w ciągu ostatniego miesiąca czułeś/aś, że nie możesz przezwyciężyć narastających trudności?",
        ],
    },
]

CUSTOM_CSS = f"""
<style>
.top-bar {{
    height: 6px;
    width: 100%;
    background: linear-gradient(to right, {COLOR_ISCO} 0%, {COLOR_ISCO} 50%, {COLOR_ESS} 50%, {COLOR_ESS} 100%);
    margin-bottom: 1.5rem;
    border-radius: 3px;
}}
.logo-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2.4rem;
    flex-wrap: wrap;
    padding: 0.2rem 0 0.9rem 0;
    margin-bottom: 0.8rem;
}}
.logo-header__item {{
    display: flex;
    justify-content: center;
    align-items: center;
    min-width: 0;
    flex: 0 1 auto;
}}
.logo-header__item img {{
    display: block;
    max-width: 220px;
    max-height: 90px;
    object-fit: contain;
}}
.logo-header__item--umcs img {{
    max-width: 220px;
    max-height: 90px;
}}
.logo-header__item--ifis img {{
    max-width: 220px;
    max-height: 90px;
}}
.logo-header__item--lodz img {{
    max-width: 220px;
    max-height: 90px;
}}
.logo-header__item--ess img {{
    max-width: 220px;
    max-height: 90px;
}}
.login-panel {{
    max-width: 420px;
    margin: 1.2rem auto 0 auto;
}}
.app-header {{
    text-align: center;
    padding: 0.5rem 0 1.5rem 0;
}}
.app-header h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {COLOR_ISCO};
    margin-bottom: 0.2rem;
}}
.app-header p {{
    font-size: 1.05rem;
    color: #444;
    margin: 0;
}}
.app-header hr {{
    border: none;
    border-top: 2px solid {COLOR_ISCO};
    width: 60%;
    margin: 0.8rem auto;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px;
}}
div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
    height: 100%;
}}
div[data-testid="column"] {{
    display: flex;
}}
div[data-testid="column"] > div {{
    width: 100%;
    display: flex;
}}
.module-card-title {{
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    line-height: 1.35;
    margin-bottom: 1rem;
    min-height: 6.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    width: 100%;
}}
.module-card-sub {{
    display: block;
    font-size: 1.0rem;
    font-weight: 600;
    color: #555;
    margin-top: 0.2rem;
}}
div[class*="st-key-pa_top10_ai_helpfulness_"][class*="_spread"] div[role="radiogroup"],
div[class*="st-key-hitl_ai_helpfulness_"][class*="_spread"] div[role="radiogroup"] {{
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
}}
div[class*="st-key-pa_top10_ai_helpfulness_"][class*="_spread"] div[role="radiogroup"] label,
div[class*="st-key-hitl_ai_helpfulness_"][class*="_spread"] div[role="radiogroup"] label {{
    flex: 1 1 0;
    justify-content: center;
}}
div[class*="st-key-questionnaire_"] div[role="radiogroup"] {{
    width: 100%;
    display: flex;
    justify-content: space-around;
    gap: 0.15rem;
}}
div[class*="st-key-questionnaire_"] div[role="radiogroup"] label {{
    flex: 1 1 0;
    justify-content: center;
    min-width: 0;
    padding: 0.25rem 0.1rem;
}}
div[class*="st-key-questionnaire_"] div[role="radiogroup"] label p {{
    display: none;
}}
div[class*="st-key-questionnaire_"] div[role="radiogroup"] label > div:first-child {{
    margin: 0 auto;
}}
div[class*="st-key-questionnaire_choice_"] button {{
    min-height: 2.5rem;
    padding: 0;
    border: 0;
    background: transparent;
    box-shadow: none;
    color: #475569;
    font-size: 1.55rem;
    line-height: 1;
}}
div[class*="st-key-questionnaire_choice_"] button:hover {{
    border: 0;
    background: #eef3f8;
    color: #003B73;
}}
div[class*="st-key-questionnaire_choice_"] button:focus {{
    box-shadow: 0 0 0 2px rgba(0, 59, 115, 0.25);
}}
div[class*="st-key-questionnaire_table_"] div[data-testid="stColumn"] {{
    border-right: 1px solid #d7dee8;
}}
div[class*="st-key-questionnaire_table_"] div[data-testid="stColumn"]:last-child {{
    border-right: 0;
}}
.questionnaire-table-head {{
    min-height: 6.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #334155;
    padding: 0.45rem 0.25rem;
    text-align: center;
    font-size: 0.78rem;
    line-height: 1.2;
    background: #eef3f8;
    border-bottom: 2px solid #94a3b8;
    margin-bottom: 0.15rem;
}}
.questionnaire-table-head--question {{
    align-items: flex-start;
    padding-left: 0.65rem;
    font-size: 0.92rem;
}}
.questionnaire-table-head__number {{
    display: block;
    color: #003B73;
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}}
@media (max-width: 700px) {{
    .logo-header {{
        gap: 1rem;
        justify-content: flex-start;
        overflow-x: auto;
        padding: 0.2rem 0 0.9rem 0;
    }}
    .logo-header__item img,
    .logo-header__item--umcs img,
    .logo-header__item--ifis img,
    .logo-header__item--lodz img,
    .logo-header__item--ess img {{
        max-width: 150px;
        max-height: 60px;
    }}
}}
</style>
"""


def _asset_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "svg+xml" if suffix == "svg" else suffix
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def render_logo_header():
    missing = [name for name, path in LOGO_PATHS.items() if not path.exists()]
    if missing:
        st.warning("Brak plików logo: " + ", ".join(missing))
        return

    st.markdown(
        f"""
        <div class="logo-header">
            <div class="logo-header__item logo-header__item--umcs">
                <img src="{_asset_data_uri(LOGO_PATHS["UMCS"])}" alt="UMCS">
            </div>
            <div class="logo-header__item logo-header__item--ifis">
                <img src="{_asset_data_uri(LOGO_PATHS["IFiS PAN"])}" alt="IFiS PAN">
            </div>
            <div class="logo-header__item logo-header__item--lodz">
                <img src="{_asset_data_uri(LOGO_PATHS["IFiS PAN Łódź"])}" alt="IFiS PAN Łódź">
            </div>
            <div class="logo-header__item logo-header__item--ess">
                <img src="{_asset_data_uri(LOGO_PATHS["ESS"])}" alt="ESS">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _valid_login(username: str, password: str) -> bool:
    expected_password = APP_USERS.get(username)
    return expected_password is not None and hmac.compare_digest(password, expected_password)


def render_helpfulness_scale(label: str, key: str) -> int:
    """Render the standard Streamlit radio scale, spread across the row via CSS.

    `index` jest podawany TYLKO gdy klucz nie ma jeszcze wartości w
    session_state - inaczej Streamlit ostrzega (i teoretycznie mógłby się
    pogubić), że wartość domyślna i session_state ustawiają widget
    jednocześnie. Dotyczy to zwłaszcza przypadków, gdzie wartość została
    wcześniej odtworzona z zapisanego szkicu (patrz _restore_widget_drafts)."""
    with st.container(key=f"{key}_spread"):
        return st.radio(
            label,
            options=[1, 2, 3, 4, 5],
            index=None if key in st.session_state else 2,
            horizontal=True,
            key=key,
        )


def render_login():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()
    st.markdown(
        """
        <div class="app-header">
            <h1>System wspomagania klasyfikacji zawodów ISCO-08</h1>
            <p>Logowanie do aplikacji</p>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-panel">', unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Użytkownik")
        password = st.text_input("Hasło", type="password")
        submitted = st.form_submit_button("Zaloguj", use_container_width=True)

    if submitted:
        if _valid_login(username.strip(), password):
            st.session_state.authenticated = True
            st.session_state.username = username.strip()
            # Zapisujemy użytkownika w URL, żeby require_login mógł go odtworzyć
            # po odświeżeniu strony bez ponownego logowania (patrz require_login).
            st.query_params["u"] = username.strip()
            st.rerun()
        else:
            st.error("Nieprawidłowy użytkownik lub hasło.")
    st.markdown("</div>", unsafe_allow_html=True)


def require_login():
    if not st.session_state.get("authenticated", False):
        # Po odświeżeniu strony (F5) st.session_state jest puste, ale identyfikator
        # użytkownika w URL (?u=...) przetrwa - odtwarzamy zalogowanie na jego
        # podstawie, żeby koder nie musiał wpisywać hasła po każdym odświeżeniu.
        qp_user = st.query_params.get("u")
        if qp_user in APP_USERS:
            st.session_state.authenticated = True
            st.session_state.username = qp_user
        else:
            render_login()
            st.stop()


def _session_cache_path(df_state_key: str) -> Path:
    """Ścieżka do pliku cache dla danego modułu (manual_df / hitl_df / hitl1d_df)
    i zalogowanego użytkownika - jeden plik na kombinację user+moduł."""
    username = st.session_state.get("username", "anon")
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_CACHE_DIR / f"{username}_{df_state_key}.pkl"


def _persist_df(df_state_key: str, df: pd.DataFrame, source_name: Optional[str] = None) -> None:
    """Zapisuje df jednocześnie do session_state (jak dotychczas) i na dysk,
    żeby przetrwał odświeżenie strony. Wywołuj zamiast gołego
    `st.session_state[df_state_key] = df` wszędzie tam, gdzie df jest
    aktualizowany po zapisaniu odpowiedzi kodera."""
    st.session_state[df_state_key] = df
    if source_name is None:
        source_key = df_state_key.replace("_df", "_source")
        source_name = st.session_state.get(source_key, "")
    try:
        path = _session_cache_path(df_state_key)
        df.to_pickle(path)
        path.with_suffix(".meta").write_text(source_name, encoding="utf-8")
    except OSError:
        # Cache na dysku to funkcja pomocnicza (odtwarzanie po F5) - jeśli zapis
        # się nie uda (np. brak miejsca), nie chcemy przerywać właściwego
        # zapisu odpowiedzi kodera, który już trafił do CODING_DB_PATH.
        pass


def _load_df_cache(df_state_key: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Odczytuje zapisany wcześniej df dla bieżącego użytkownika, jeśli istnieje."""
    path = _session_cache_path(df_state_key)
    meta_path = path.with_suffix(".meta")
    if path.exists() and meta_path.exists():
        try:
            return pd.read_pickle(path), meta_path.read_text(encoding="utf-8")
        except (OSError, ValueError, EOFError):
            return None, None
    return None, None


# Krótkie nazwy parametrów URL dla poszczególnych idx_state_key - identyczny
# mechanizm jak przy zapamiętywaniu zalogowanego użytkownika (?u=) i aktualnej
# strony (?p=), tylko dla dokładnej pozycji kodowania w danym module.
IDX_QUERY_PARAM = {
    "manual_idx": "mi",
    "hitl_idx": "hi",
    "hitl1d_idx": "h1i",
}
FRONTIER_QUERY_PARAM = {
    "manual_idx": "mf",
    "hitl_idx": "hf",
    "hitl1d_idx": "h1f",
}


def _set_idx(idx_state_key: str, value: int) -> None:
    """Ustawia bieżącą pozycję kodowania jednocześnie w session_state i w URL,
    żeby DOKŁADNA pozycja (np. przypadek nr 99) przetrwała odświeżenie strony
    (F5) - zamiast cofać się do pierwszego nieukończonego przypadku, co mogłoby
    być inną pozycją, jeśli koder wcześniej przeskakiwał między przypadkami
    (patrz _render_case_jumper). Dodatkowo śledzi "frontier" - najdalej
    osiągniętą pozycję w danym module - używane przez _next_idx_after_save,
    żeby po edycji wcześniejszego przypadku wrócić tam, gdzie koder naprawdę
    skończył, zamiast przesuwać się tylko o jeden dalej od edytowanego miejsca.
    Używaj zamiast gołego `st.session_state[idx_state_key] = wartość`."""
    st.session_state[idx_state_key] = value
    frontier_key = _frontier_key(idx_state_key)
    frontier = max(value, st.session_state.get(frontier_key, value))
    st.session_state[frontier_key] = frontier
    param = IDX_QUERY_PARAM.get(idx_state_key)
    if param:
        st.query_params[param] = str(value)
    frontier_param = FRONTIER_QUERY_PARAM.get(idx_state_key)
    if frontier_param:
        st.query_params[frontier_param] = str(frontier)


def _frontier_key(idx_state_key: str) -> str:
    return f"{idx_state_key}_frontier"


def _restore_frontier_from_query(idx_state_key: str, fallback: int) -> int:
    """Odczytuje zapamiętaną najdalej osiągniętą pozycję z URL (patrz
    _set_idx) po odświeżeniu strony. Nigdy nie zwraca wartości mniejszej niż
    `fallback`, żeby po refreshu frontier nie "cofnął się" poniżej pozycji,
    do której i tak wracamy (np. pierwszy nieukończony przypadek)."""
    param = FRONTIER_QUERY_PARAM.get(idx_state_key)
    if not param:
        return fallback
    raw = st.query_params.get(param)
    if raw is None:
        return fallback
    try:
        return max(int(raw), fallback)
    except ValueError:
        return fallback


def _next_idx_after_save(qualifying_positions: list[int], idx: int, idx_state_key: str, n: int) -> int:
    """Po zapisaniu odpowiedzi decyduje, dokąd przejść dalej.

    Jeśli koder edytował przypadek PRZED swoją najdalej osiągniętą pozycją -
    czyli wrócił, żeby coś sprawdzić albo poprawić (patrz _render_case_jumper
    i przyciski "Poprzedni") - po zapisaniu WRACA na tę najdalszą pozycję,
    zamiast przesuwać się tylko o jeden przypadek od właśnie edytowanego
    miejsca (co wyglądałoby jak "nic się nie zapisało", bo koder gubił swoje
    właściwe miejsce w pliku). W przeciwnym razie - normalny postęp "do przodu"
    - przechodzi po prostu do kolejnego kwalifikującego się przypadku, jak
    dotychczas."""
    next_seq = _next_qualifying_idx(qualifying_positions, idx, n)
    frontier = st.session_state.get(_frontier_key(idx_state_key), idx)
    if idx < frontier:
        return frontier
    return next_seq


def _restore_idx_from_query(idx_state_key: str, fallback: int, n: int) -> int:
    """Odczytuje zapamiętaną pozycję z URL (patrz _set_idx) po odświeżeniu
    strony. Jeśli parametru brak albo jest spoza zakresu pliku, używa
    `fallback` (zwykle pierwszy nieukończony przypadek - patrz
    _first_unfinished_idx)."""
    param = IDX_QUERY_PARAM.get(idx_state_key)
    if not param:
        return fallback
    raw = st.query_params.get(param)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    if 0 <= value <= n:
        return value
    return fallback


# ============================================================
# ZASOBY (cache - wczytywane raz na sesję serwera)
# ============================================================
class OllamaEmbedder:
    """Cienki wrapper na lokalne API Ollama, naśladujący interfejs
    SentenceTransformer.encode() używany w reszcie kodu poniżej."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = OLLAMA_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = ollama.embed(model=self.model_name, input=batch)
            all_embeddings.extend(response.embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings


@st.cache_resource(show_spinner=False)
def load_model():
    return OllamaEmbedder()


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Wczytuje zapisane wcześniej embeddingi .npy + metadane. Brak ChromaDB."""
    emb_path = Path(EMB_DIR)

    title_emb = np.load(emb_path / "title_emb.npy")
    tasks_emb = np.load(emb_path / "tasks_emb.npy")
    synteza_emb = np.load(emb_path / "synteza_emb.npy")

    with open(emb_path / "codes_ordered.json", encoding="utf-8") as f:
        codes_ordered = json.load(f)

    with open(emb_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return title_emb, tasks_emb, synteza_emb, codes_ordered, metadata


@st.cache_resource(show_spinner=False)
def load_embeddings_level(level: int):
    """Wczytuje embeddingi .npy + metadane dla pojedynczego poziomu hierarchii ISCO-08
    (1 = grupy główne ... 4 = grupy elementarne), używane w trybie kaskadowym."""
    emb_path = Path(LEVEL_EMB_DIRS[level])

    title_emb = np.load(emb_path / "title_emb.npy")
    tasks_emb = np.load(emb_path / "tasks_emb.npy")
    synteza_emb = np.load(emb_path / "synteza_emb.npy")

    with open(emb_path / "codes_ordered.json", encoding="utf-8") as f:
        codes_ordered = json.load(f)

    with open(emb_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return title_emb, tasks_emb, synteza_emb, codes_ordered, metadata


VAR_METADATA_PATHS = {
    "Respondent": APP_DIR / "ess_var_metadata_pl_respondent.json",
    "Partner": APP_DIR / "ess_var_metadata_pl_partner.json",
}


def load_var_metadata(target: str = "Respondent") -> dict:
    """Wczytuje metadane zmiennych tabelarycznych (etykiety + etykiety wartości)
    wyeksportowane z pliku ESS .RData (patrz extract_metadata.R) - OSOBNO dla
    respondenta i dla partnera (dwa różne pliki, dwa różne zestawy zmiennych).
    Dzięki temu w trybie 'Respondent' dymki/podpowiedzi pokazują wyłącznie
    zmienne respondenta, a w trybie 'Partner' - wyłącznie zmienne partnera.
    Zwraca pusty słownik, jeśli plik nie istnieje - reszta kodu ma to
    obsłużone bez błędów.

    UWAGA: celowo BEZ @st.cache_resource - to mały plik JSON, tani w odczycie,
    a cache_resource trzymałby wynik (w tym pusty słownik, gdy plik jeszcze
    nie istniał) na stałe między rerunami, mimo późniejszej zmiany plików
    albo dodania pliku na dysk."""
    path = Path(VAR_METADATA_PATHS.get(target, VAR_METADATA_PATHS["Respondent"]))
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mode_var_names(target: str) -> set:
    """Zbiór nazw kolumn należących do danego trybu: zmienne z odpowiedniego
    pliku metadanych (ess_var_metadata_pl_respondent/partner.json) plus
    kolumny zawód/obowiązki/wykształcenie tego trybu (C30-C32 albo C45-C47).
    Też celowo bez cache - patrz komentarz w load_var_metadata."""
    names = set(load_var_metadata(target).keys())
    names |= set(TARGET_COLUMNS[target].values())
    return names


def _warn_if_meta_missing(target: str):
    """Pokazuje krótkie ostrzeżenie, jeśli plik metadanych zmiennych dla
    danego trybu nie został znaleziony w folderze roboczym - pomaga od razu
    zdiagnozować brak dymków (tooltipów) w tabeli, zamiast zgadywać.

    UWAGA: celowo pokazuje tylko nazwę pliku i nazwę folderu roboczego (np.
    "App_2Mod_RP2"), NIGDY pełnej ścieżki bezwzględnej (np.
    /Users/imie_nazwisko/Desktop/...) - to lokalna ścieżka na dysku kodera,
    nie powinna się pojawiać w interfejsie."""
    path = Path(VAR_METADATA_PATHS.get(target, VAR_METADATA_PATHS["Respondent"]))
    if not path.exists():
        st.caption(
            f"⚠️ Brak pliku metadanych `{path.name}` w folderze roboczym "
            f"(`{Path.cwd().name}`) - dymki (opisy zmiennych) będą niedostępne."
        )


def visible_df_for_mode(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Zwraca df zawężony WYŁĄCZNIE do kolumn należących do bieżącego trybu
    (whitelist, nie blacklist): w trybie 'Respondent' widać tylko zmienne
    respondenta (z ess_var_metadata_pl_respondent.json), w trybie 'Partner' -
    tylko zmienne partnera (z ess_var_metadata_pl_partner.json). Wszystko
    inne (kolumny drugiego trybu, kolumny wynikowe typu ISCO_wybrany,
    nieznane/niesklasyfikowane kolumny) jest ukryte.

    Kolumny zawód/obowiązki/wykształcenie (B33-B35 albo B48-B50) też są
    ukryte z tabeli - są już pokazane osobno w ramce pod tabelą (patrz
    render_classify_hitl / render_classify_hitl_1digit), więc w tabeli
    tylko by się powtarzały."""
    allowed = _mode_var_names(target) - set(TARGET_COLUMNS[target].values())
    keep_cols = [c for c in df.columns if c in allowed]

    # ID respondenta zawsze widoczne w tabeli jako pierwsza kolumna, niezależnie
    # od whitelisty trybu (idno nie jest zmienną z pliku metadanych, więc bez
    # tego byłoby ukryte - a koderzy chcą je widzieć wprost w tabelce, nie
    # tylko w komunikacie nad nią).
    idno_col = next((c for c in df.columns if str(c).strip().lower() == "idno"), None)
    if idno_col is not None and idno_col not in keep_cols:
        keep_cols = [idno_col] + keep_cols

    return df[keep_cols]


# Ręczne etykiety dla kolumn, których opis w metadanych z pliku ESS (.RData)
# albo nie istnieje, albo jest zbyt techniczny - nadrzędne wobec var_meta.
# Działa niezależnie od zawartości pliku ess_var_metadata_pl_*.json.
COLUMN_LABEL_OVERRIDES = {
    "B31": "Branża",
}


def _build_column_help(col: str, var_meta: dict) -> Optional[str]:
    """Buduje tekst dymka (tooltip) dla nagłówka kolumny na podstawie metadanych
    zmiennej: etykieta zmiennej + (jeśli jest ich rozsądnie mało) lista etykiet
    wartości. Zwraca None, jeśli brak metadanych dla tej kolumny."""
    meta = var_meta.get(col) or {}
    label = COLUMN_LABEL_OVERRIDES.get(col, meta.get("label", ""))
    value_labels = meta.get("value_labels", {}) or {}

    parts = []
    if label:
        parts.append(label)

    if value_labels:
        vl_lines = [f"{k} = {v}" for k, v in value_labels.items()]
        parts.append("\n".join(vl_lines))

    return "\n\n".join(parts) if parts else None


def build_column_config(df: pd.DataFrame, var_meta: dict) -> dict:
    """Buduje słownik column_config dla st.dataframe, żeby po najechaniu na
    nagłówek kolumny pokazywał się dymek z opisem zmiennej (i etykietami
    wartości, jeśli jest ich niedużo)."""
    config = {}
    for col in df.columns:
        help_text = _build_column_help(col, var_meta)
        if help_text:
            config[col] = st.column_config.Column(help=help_text)
    return config


MAX_LABEL_LINE_LEN = 160  # limit znaków w głównej linii etykiety (kod + nazwa PL +
# " - dopasowanie: x.xxx") - podniesiony razem ze zmianą layoutu strony na "wide" (więcej
# miejsca poziomego), więc skracanie wielokropkiem powinno teraz być rzadkością, nie regułą


def _format_candidate_label(isco_code: str, title_pl: str, title_en: str = "", score: Optional[float] = None) -> str:
    """Buduje etykietę kandydata do widgetu wyboru (st.radio):
    '<kod> — <nazwa PL> - dopasowanie: <wartość>' + (jeśli dostępny)
    angielski odpowiednik w osobnym akapicie pod spodem, kursywą - jako
    najbliższe dostępne przybliżenie "mniejszej czcionki" w zwykłym tekście
    opcji (st.radio nie obsługuje HTML/CSS ani realnego rozmiaru fontu w
    opcjach, tylko podstawowy markdown, jeśli w ogóle). To PRÓBA wymuszenia
    złamania linii wewnątrz jednej opcji - poprzedni test z pojedynczym \\n
    sklejał się w jedną linię, dlatego tu używamy podwójnego \\n\\n (znak
    akapitu w markdown) - jeśli to również się nie uda, jedynym pewnym
    rozwiązaniem pozostaje pokazanie angielskiego odpowiednika w osobnym,
    zwykłym bloku tekstu nad listą (poza opcjami radio).

    Główna linia (kod — nazwa PL - dopasowanie) jest ograniczona do
    MAX_LABEL_LINE_LEN znaków - jeśli polska nazwa zawodu jest zbyt długa,
    zostaje przycięta i zakończona wielokropkiem "…", żeby ta linia NIGDY
    nie zawijała się na dwie, niezależnie od długości nazwy czy szerokości
    ekranu. Angielski odpowiednik w drugim akapicie NIE jest przycinany -
    to osobna linia, więc jej ewentualne zawinięcie nie psuje głównej."""
    title_display = _lower_first(title_pl)
    suffix = f" - dopasowanie: {score:.3f}" if score is not None else ""
    prefix = f"**{isco_code}** — "
    # Liczymy budżet znaków bez markdownowych gwiazdek (**), żeby pogrubienie
    # nie skracało realnie dostępnego miejsca na nazwę zawodu.
    budget = MAX_LABEL_LINE_LEN - (len(prefix) - 4) - len(suffix)
    if budget > 10 and len(title_display) > budget:
        title_display = title_display[: budget - 1].rstrip(" ,;.-") + "…"
    label = prefix + title_display + suffix
    if title_en:
        label += f"\n\n*(ang. {title_en})*"
    return label


def _lower_first(s: str) -> str:
    """Zamienia pierwszą literę tekstu na małą (do wyświetlania nazw zawodów
    ISCO-08 - w oficjalnych tytułach zaczynają się wielką literą, a chcemy
    małą przy prezentacji w apce).

    UWAGA: tytuły poziomu 1 (główne grupy) są w Excelu zapisane CAŁYMI
    WIELKIMI LITERAMI (np. "PRACOWNICY USŁUG I SPRZEDAWCY"), w odróżnieniu
    od poziomów 2-4, które mają zwykłą "wielka litera na początku zdania".
    Jeśli tego nie obsłużymy, zamiana samej pierwszej litery zostawia resztę
    wielkimi literami (np. "pRACOWNICY USŁUG I SPRZEDAWCY"), dlatego
    najpierw normalizujemy cały-wielkimi-literami tekst do zwykłej postaci.
    """
    if not s:
        return s
    if s.isupper():
        s = s[0] + s[1:].lower()
    return s[0].lower() + s[1:]


def _mode_key(df_state_key: str) -> str:
    """Klucz session_state przechowujący tryb kodowania (Respondent/Partner)
    dla CAŁEGO wczytanego pliku w danym module (df_state_key: 'hitl_df'
    albo 'hitl1d_df') - nie per respondent, tylko jeden globalny wybór."""
    return f"coding_mode_{df_state_key}"


def _get_coding_target(df_state_key: str) -> str:
    """Zwraca aktualnie wybrany tryb kodowania ('Respondent' albo 'Partner')
    dla całego pliku w danym module - domyślnie 'Respondent'."""
    return st.session_state.get(_mode_key(df_state_key), "Respondent")


def _target_cols(df_state_key: str) -> dict:
    """Zwraca słownik {'zawod': ..., 'obowiazki': ..., 'wyksztalcenie': ...}
    z nazwami kolumn odpowiadającymi aktualnie wybranemu trybowi."""
    return TARGET_COLUMNS[_get_coding_target(df_state_key)]


def _ensure_text_column_dtype(df: pd.DataFrame, col: str) -> None:
    """Wymusza dtype 'object' na kolumnie, która ma przechowywać tekst/kody
    ISCO-08 (a nie liczby) - w miejscu, bez tworzenia nowego df.

    Zapobiega błędowi pandas "Invalid value '...' for dtype 'float64'", który
    pojawia się przy WZNOWIENIU kodowania z wcześniej częściowo wypełnionego
    pliku CSV: skoro istniejące kody ISCO-08 w takiej kolumnie wyglądają jak
    liczby (np. "1420" zapisane bez cudzysłowu w CSV), pandas przy wczytaniu
    automatycznie nadaje całej kolumnie typ float64 - a wtedy próba zapisania
    KOLEJNEJ wartości jako zwykły string (np. nowo wybranego kodu) wywala
    wyjątek zamiast po cichu przekonwertować typ.

    Istniejące wartości liczbowe w stylu 1420.0 są przy okazji sprowadzane
    z powrotem do czystego stringa "1420" (bez zbędnego ".0"), żeby stare i
    nowo dopisywane wiersze miały spójny format w eksportowanym pliku."""
    if col not in df.columns or df[col].dtype == "object":
        return

    def _to_text(v):
        if pd.isna(v):
            return None
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    df[col] = df[col].map(_to_text).astype("object")


def _ensure_object_dtype(df: pd.DataFrame, col: str) -> None:
    """Wymusza dtype 'object' na kolumnie BEZ zmiany samych wartości - patrz
    _ensure_text_column_dtype. Używane dla kolumn, które nie są kodami/tekstem
    (np. bool "Czy_uzytkownik_wracal"), więc nie chcemy stringować wartości,
    tylko dopuścić dowolny typ przy kolejnych zapisach."""
    if col in df.columns and df[col].dtype != "object":
        df[col] = df[col].astype("object")


def _display_respondent_idno(row: pd.Series) -> None:
    """Pokazuje IDNO niezależnie od jego wielkości liter i typu z CSV."""
    idno_col = next((col for col in row.index if str(col).strip().lower() == "idno"), None)
    if idno_col is None or pd.isna(row[idno_col]):
        st.warning("Brak numeru IDNO dla tego respondenta.")
        return

    value = row[idno_col]
    if isinstance(value, float) and value.is_integer():
        value = str(int(value))
    else:
        value = str(value).strip()
    st.info(f"**IDNO respondenta: `{value}`**")


QUALIFYING_FLAG_COLUMNS = {
    "Respondent": "Respondent_analiza",
    "Partner": "Partner_analiza",
}


def _qualifying_mask(df: pd.DataFrame, target: str) -> pd.Series:
    """Zwraca maskę bool (per wiersz) wskazującą, czy dany wiersz W OGÓLE
    powinien trafić do kodowania w trybie `target` ('Respondent' albo
    'Partner'). Sprawdzana jest flaga w kolumnie
    QUALIFYING_FLAG_COLUMNS[target] ('Respondent_analiza' dla Respondentów,
    'Partner_analiza' dla Partnerów) - wiersz kwalifikuje się, gdy ta
    flaga == 1 (obsługiwane formaty: liczba 1, 1.0, string "1"). Wartości
    puste, 0 albo cokolwiek innego oznaczają pominięcie wiersza. Jeśli w
    pliku nie ma takiej kolumny wcale, WSZYSTKIE wiersze się kwalifikują
    (kompatybilność wsteczna ze starszymi plikami bez tych flag)."""
    col = QUALIFYING_FLAG_COLUMNS.get(target)
    if not col or col not in df.columns:
        return pd.Series(True, index=df.index)

    def _is_one(v):
        if pd.isna(v):
            return False
        try:
            return float(v) == 1.0
        except (TypeError, ValueError):
            return str(v).strip() == "1"

    return df[col].apply(_is_one)


def _qualifying_positions(df: pd.DataFrame, target: str) -> list[int]:
    """Zwraca posortowaną listę pozycji (0-indexed, zgodnych z df.iloc/df.at)
    wierszy kwalifikujących się do kodowania w trybie `target` - patrz
    _qualifying_mask. Wiersze niekwalifikujące się są całkowicie pomijane w
    nawigacji (nigdy nie są pokazywane koderowi)."""
    mask = _qualifying_mask(df, target).to_numpy()
    return [i for i, ok in enumerate(mask) if ok]


def _next_qualifying_idx(qualifying_positions: list[int], current_idx: int, n: int) -> int:
    """Zwraca najbliższą kwalifikującą się pozycję ŚCIŚLE większą niż
    current_idx, albo n (koniec/zakończono), jeśli żadna dalsza się nie
    kwalifikuje. Używane zamiast zwykłego "idx + 1" przy przechodzeniu do
    kolejnej osoby, żeby pomijać wiersze niespełniające flagi
    Respondent_analiza / Partner_analiza."""
    for pos in qualifying_positions:
        if pos > current_idx:
            return pos
    return n


def _prev_qualifying_idx(qualifying_positions: list[int], current_idx: int) -> int:
    """Zwraca najbliższą kwalifikującą się pozycję ŚCIŚLE mniejszą niż
    current_idx. Jeśli żadna wcześniejsza się nie kwalifikuje, zwraca
    current_idx bez zmian (nie ma dokąd się cofnąć - przycisk "Poprzedni"
    powinien się wtedy po prostu nie pokazywać, patrz miejsca wywołania)."""
    prev = None
    for pos in qualifying_positions:
        if pos >= current_idx:
            break
        prev = pos
    return prev if prev is not None else current_idx


def _render_case_jumper(
    qualifying_positions: list[int],
    idx: int,
    idx_state_key: str,
    key_suffix: str,
    on_jump=None,
) -> None:
    """Pozwala przeskoczyć do DOWOLNEGO przypadku na liście kwalifikujących się
    wierszy (patrz _qualifying_positions), a nie tylko o jeden w przód/w tył.

    Przeskoczenie nie kasuje żadnych wcześniej zapisanych kodów - każdy
    przypadek trzyma swoje dane niezależnie w wierszu df (df.at[idx, ...]),
    więc przejście gdzie indziej i powrót nic nie nadpisuje. Dzięki temu można
    np. wrócić 2 przypadki wstecz, żeby coś sprawdzić, albo pominąć trudniejszy
    przypadek i najpierw zrobić łatwiejsze dalej na liście, a wrócić do niego
    później.

    `on_jump`, jeśli podane, jest wywoływane z docelowym idx PRZED zmianą
    st.session_state[idx_state_key] - używane np. w module ręcznym do
    zresetowania wizarda kaskady dla docelowego przypadku (patrz
    _manual_reset_idx)."""
    total = len(qualifying_positions)
    if total <= 1:
        return
    current_rank = qualifying_positions.index(idx) + 1 if idx in qualifying_positions else 1
    with st.expander(f"Przejdź do przypadku (aktualnie {current_rank} z {total})"):
        col_num, col_btn = st.columns([3, 1])
        with col_num:
            target_rank = st.number_input(
                "Numer przypadku",
                min_value=1,
                max_value=total,
                value=current_rank,
                step=1,
                key=f"jump_rank_{key_suffix}",
                label_visibility="collapsed",
            )
        with col_btn:
            if st.button("Przejdź", key=f"jump_btn_{key_suffix}", use_container_width=True):
                target_idx = qualifying_positions[int(target_rank) - 1]
                if on_jump is not None:
                    on_jump(target_idx)
                _set_idx(idx_state_key, target_idx)
                st.rerun()


def _render_next_case_button(qualifying_positions: list[int], idx: int, idx_state_key: str, key_suffix: str, on_jump=None) -> bool:
    """Przycisk 'Przejdź do kolejnego przypadku' - przesuwa o jedną pozycję do
    przodu na liście kwalifikujących się wierszy, bez zapisywania niczego
    (czysta nawigacja, jak _render_case_jumper). Umieszczany OBOK przycisku
    'Poprzedni'/'Wróć do poprzedniego przypadku', a nie w schowanym panelu, dla
    szybkiego, jednoklikowego przeglądu do przodu. Zwraca True, jeśli przycisk
    został kliknięty (wtedy wywołujący powinien od razu zrobić st.rerun())."""
    current_rank = qualifying_positions.index(idx) + 1 if idx in qualifying_positions else 0
    if current_rank >= len(qualifying_positions):
        return False
    if st.button("Przejdź do kolejnego przypadku →", key=f"jump_next_{key_suffix}", use_container_width=True):
        target_idx = qualifying_positions[current_rank]
        if on_jump is not None:
            on_jump(target_idx)
        _set_idx(idx_state_key, target_idx)
        return True
    return False


def _qualifying_progress(qualifying_positions: list[int], idx: int, n: int) -> tuple[float, int, int]:
    """Zwraca (ułamek_postępu, aktualna_pozycja_1_indexed, łączna_liczba) do
    wyświetlenia na pasku postępu, licząc WYŁĄCZNIE kwalifikujące się wiersze
    (patrz _qualifying_positions) - a nie surową liczbę wszystkich wierszy w
    pliku, skoro część z nich jest pomijana (Respondent_analiza /
    Partner_analiza != 1)."""
    total = len(qualifying_positions)
    if total == 0:
        return 0.0, 0, 0
    if idx >= n:
        completed = total
    else:
        completed = qualifying_positions.index(idx) if idx in qualifying_positions else 0
    fraction = completed / total
    current_rank = min(completed + 1, total)
    return fraction, current_rank, total


def _first_unfinished_idx(df: pd.DataFrame, target: str) -> int:
    """Zwraca indeks pierwszego KWALIFIKUJĄCEGO SIĘ wiersza (patrz
    _qualifying_mask - flaga Respondent_analiza / Partner_analiza == 1),
    dla którego NIE zapisano jeszcze decyzji kodera w trybie `target`
    ('Respondent' albo 'Partner') - czyli miejsca, od którego trzeba
    (kontynuować) kodowanie. Wiersz uznajemy za już zakodowany, gdy jego
    'Kodowany_podmiot' zgadza się z `target` ORAZ wypełniony jest
    'ISCO_wybrany' albo 'Brak_mozliwosci_zakodowania' == "Tak". Wiersze
    NIEkwalifikujące się są traktowane jak już gotowe (pomijane) - nigdy nie
    są pokazywane koderowi. Jeśli wszystkie kwalifikujące się wiersze mają
    już decyzję (albo nie ma żadnych kwalifikujących się w ogóle), zwraca
    len(df) (koniec pliku - kodowanie w tym trybie jest kompletne).

    Dzięki temu, jeśli koder przerwie sesję w połowie (np. po 10 osobach) i
    wróci później do TEGO SAMEGO, częściowo wypełnionego pliku - czy to w tej
    samej, czy w zupełnie nowej sesji przeglądarki (po ponownym wgraniu
    wcześniej pobranego CSV) - aplikacja sama wznowi kodowanie od pierwszej
    nieoznaczonej osoby, zamiast zaczynać od zera. Dla zupełnie świeżego pliku
    (bez żadnych decyzji zapisanych) zwraca indeks pierwszego kwalifikującego
    się wiersza (0, jeśli ten się kwalifikuje)."""
    qualifies = _qualifying_mask(df, target)
    if "Kodowany_podmiot" not in df.columns:
        positions = qualifies.to_numpy().nonzero()[0]
        return int(positions[0]) if len(positions) else len(df)
    decided = df["Kodowany_podmiot"] == target
    if "ISCO_wybrany" in df.columns:
        decided = decided & df["ISCO_wybrany"].notna()
    else:
        decided = decided & False
    if "Brak_mozliwosci_zakodowania" in df.columns:
        decided = decided | ((df["Kodowany_podmiot"] == target) & (df["Brak_mozliwosci_zakodowania"] == "Tak"))
    # Wiersze niekwalifikujące się traktujemy jako "gotowe" (pomijane) - nie
    # mają być pokazywane koderowi w ogóle.
    decided = decided | (~qualifies)
    undecided_positions = (~decided).to_numpy().nonzero()[0]
    return int(undecided_positions[0]) if len(undecided_positions) else len(df)


def _unfinished_case_numbers(df: pd.DataFrame, target: str, qualifying_positions: list[int]) -> list[int]:
    """Jak _first_unfinished_idx, ale zwraca numery (rangi 1-based wśród
    `qualifying_positions`, czyli te same numery co widoczne koderowi w
    pasku postępu 'X z Y') WSZYSTKICH jeszcze nieukończonych przypadków, a
    nie tylko pierwszego - używane przy pobieraniu częściowego wyniku, żeby
    pokazać listę numerów do dokończenia."""
    qualifies = _qualifying_mask(df, target)
    if "Kodowany_podmiot" not in df.columns:
        decided = pd.Series(False, index=df.index)
    else:
        decided = df["Kodowany_podmiot"] == target
        if "ISCO_wybrany" in df.columns:
            decided = decided & df["ISCO_wybrany"].notna()
        else:
            decided = decided & False
        if "Brak_mozliwosci_zakodowania" in df.columns:
            decided = decided | ((df["Kodowany_podmiot"] == target) & (df["Brak_mozliwosci_zakodowania"] == "Tak"))
    decided = decided | (~qualifies)
    undecided_idx_set = set((~decided).to_numpy().nonzero()[0].tolist())
    return [rank for rank, pos in enumerate(qualifying_positions, start=1) if pos in undecided_idx_set]


def _reset_module_progress(df_state_key: str, idx_state_key: str, df=None, target_mode: Optional[str] = None):
    """Czyści cały postęp kodowania w danym module (wybory, cache klasyfikacji,
    liczniki czasu, stan kaskady, zaznaczenia w tabeli). Używane przy twardym
    przełączeniu trybu Respondent/Partner w trakcie sesji. Same dane (df) i
    wynik zapisany w kolumnach wynikowych NIE są czyszczone - o ich pobranie
    (częściowy CSV) prosimy PRZED przełączeniem.

    Jeśli podano `df` i `target_mode`, indeks NIE wraca do zera na sztywno -
    zamiast tego, tak jak przy wznowieniu z pliku, ustawiany jest na pierwszą
    nieukończoną osobę w trybie `target_mode` (patrz _first_unfinished_idx) -
    dzięki temu powrót do drugiego trybu (np. z Respondentów na Partnerów)
    też trafia tam, gdzie koder poprzednio skończył, a nie zawsze na start."""
    clear_prefixes = ("manual_", "hitl_", "hitl1d_", "cascade_", "pa1_", "pa_top10_", "resp_table_")
    keep_keys = {
        df_state_key,
        idx_state_key,
        "manual_source", "hitl_source", "hitl1d_source",
        "hitl_source_cols", "hitl1d_source_cols",
        _mode_key(df_state_key),
    }
    for key in list(st.session_state.keys()):
        if key in keep_keys:
            continue
        if key.startswith(clear_prefixes):
            del st.session_state[key]
    if df is not None and target_mode:
        _set_idx(idx_state_key, _first_unfinished_idx(df, target_mode))
    else:
        _set_idx(idx_state_key, 0)


def render_mode_selector(df_state_key: str, idx_state_key: str, df, idx: int, n: int) -> str:
    """Renderuje dwa przyciski 'Koduj respondentów' / 'Koduj partnerów'.
    Wybór dotyczy CAŁEGO wczytanego pliku, nie tylko aktualnie wyświetlanego
    wiersza. Jeśli kodowanie w bieżącym trybie jest już W TRAKCIE (nie
    ukończono jeszcze wszystkich respondentów), kliknięcie drugiego trybu NIE
    przełącza od razu - pokazuje ostrzeżenie z możliwością pobrania
    częściowego wyniku i zakończenia bieżącej sesji. Zakończenie NIE przełącza
    automatycznie na drugi tryb - to osobna, świadoma decyzja kodera.

    W obu przypadkach (przełączenie bezpośrednie, gdy drugi tryb nie jest w
    trakcie, oraz przełączenie po "Zakończ kodowanie") indeks NIE wraca na
    sztywno do zera - ustawiany jest na pierwszą nieukończoną osobę w nowym
    trybie (patrz _first_unfinished_idx), więc powrót do drugiego trybu też
    trafia tam, gdzie koder poprzednio skończył."""
    mode_key = _mode_key(df_state_key)
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Respondent"
    current = st.session_state[mode_key]

    in_progress = 0 < idx < n
    pending_key = f"pending_mode_{df_state_key}"
    pending_target_key = f"pending_target_{df_state_key}"

    if st.session_state.get(pending_key):
        current_plural = "respondentów" if current == "Respondent" else "partnerów"
        qualifying_positions_switch = _qualifying_positions(df, current)
        _, progress_rank_switch, progress_total_switch = _qualifying_progress(qualifying_positions_switch, idx, n)

        st.warning(
            f"Kodowanie {current_plural} nie zostało jeszcze ukończone "
            f"({progress_rank_switch - 1} z {progress_total_switch}). "
            "Najpierw pobierz do CSV obecne wyniki, a potem zakończ kodowanie - "
            "inaczej niezapisany postęp zostanie utracony."
        )

        partial_csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        mode_suffix = "respondent" if current == "Respondent" else "partner"

        col_dl, col_end, col_cancel = st.columns(3)
        with col_dl:
            st.download_button(
                "Pobierz częściowy wynik (CSV)",
                data=partial_csv,
                file_name=f"wynik_czesciowy_{mode_suffix}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"partial_dl_{df_state_key}",
            )
        with col_end:
            if st.button(
                "Zakończ kodowanie",
                type="primary",
                use_container_width=True,
                key=f"end_session_{df_state_key}",
            ):
                target_mode = st.session_state.pop(pending_target_key, None)
                st.session_state.pop(pending_key, None)
                _reset_module_progress(df_state_key, idx_state_key, df=df, target_mode=target_mode)
                if target_mode:
                    st.session_state[mode_key] = target_mode
                st.rerun()
        with col_cancel:
            if st.button(
                "Anuluj, wróć do kodowania",
                use_container_width=True,
                key=f"cancel_switch_{df_state_key}",
            ):
                st.session_state.pop(pending_key, None)
                st.session_state.pop(pending_target_key, None)
                st.rerun()

        return current

    col_r, col_p = st.columns(2)
    with col_r:
        if st.button(
            "Koduj respondentów",
            use_container_width=True,
            type="primary" if current == "Respondent" else "secondary",
            key=f"mode_btn_respondent_{df_state_key}",
        ):
            if current != "Respondent":
                if in_progress:
                    st.session_state[pending_key] = True
                    st.session_state[pending_target_key] = "Respondent"
                else:
                    st.session_state[mode_key] = "Respondent"
                    _set_idx(idx_state_key, _first_unfinished_idx(df, "Respondent"))
                st.rerun()
    with col_p:
        if st.button(
            "Koduj partnerów",
            use_container_width=True,
            type="primary" if current == "Partner" else "secondary",
            key=f"mode_btn_partner_{df_state_key}",
        ):
            if current != "Partner":
                if in_progress:
                    st.session_state[pending_key] = True
                    st.session_state[pending_target_key] = "Partner"
                else:
                    st.session_state[mode_key] = "Partner"
                    _set_idx(idx_state_key, _first_unfinished_idx(df, "Partner"))
                st.rerun()

    return st.session_state[mode_key]





def _decode_value(raw_val, value_labels: dict) -> Optional[str]:
    if pd.isna(raw_val) or not value_labels:
        return None
    key_candidates = [str(raw_val)]
    try:
        key_candidates.append(str(int(float(raw_val))))
    except (ValueError, TypeError):
        pass
    for k in key_candidates:
        if k in value_labels:
            return value_labels[k]
    return None


def build_column_config_for_respondent(row, var_meta: dict) -> dict:
    """Buduje column_config dla tabeli JEDNEGO respondenta: dymek po
    najechaniu pokazuje nazwę zmiennej, jej opis, i konkretną WYBRANĄ
    wartość tego respondenta wraz z wyjaśnieniem (a nie całą listę
    wszystkich możliwych kategorii)."""
    config = {}
    for col in row.index:
        meta = var_meta.get(col) or {}
        label = COLUMN_LABEL_OVERRIDES.get(col, meta.get("label", ""))
        if not label:
            continue
        value_labels = meta.get("value_labels", {}) or {}
        raw_val = row.get(col)
        decoded = _decode_value(raw_val, value_labels)

        parts = [label]
        if decoded:
            parts.append(f"Wybrana wartość: {raw_val} = {decoded}")
        else:
            parts.append(f"Wartość: {raw_val}")

        config[col] = st.column_config.Column(help="\n\n".join(parts))
    return config


# ============================================================
# LOGIKA KLASYFIKACJI (czysty numpy, bez bazy wektorowej)
# ============================================================
def classify(
    zawod_czlowieka: str,
    umiejetnosci_obowiazki: str,
    wyksztalcenie: str,
    model,
    title_emb: np.ndarray,
    tasks_emb: np.ndarray,
    synteza_emb: np.ndarray,
    codes_ordered: list,
    metadata: dict,
    top_k: int = 5,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    q_zawod = model.encode([QUERY_INSTRUCTION + zawod_czlowieka], normalize_embeddings=True)
    q_skills = model.encode([QUERY_INSTRUCTION + umiejetnosci_obowiazki], normalize_embeddings=True)

    # Embeddingi są znormalizowane, więc inner product = cosine similarity.
    # S1 = B33 (nazwa zawodu) <-> Nazwa      | waga WEIGHTS["title"]
    # S2 = B34 (zadania)      <-> Zadania    | waga WEIGHTS["tasks"]
    # S3 = B34 (zadania)      <-> Synteza    | waga WEIGHTS["synteza"]
    sim_title = (title_emb @ q_zawod.T).flatten()
    sim_tasks = (tasks_emb @ q_skills.T).flatten()
    sim_synteza = (synteza_emb @ q_skills.T).flatten()

    score = (
        WEIGHTS["title"] * sim_title
        + WEIGHTS["tasks"] * sim_tasks
        + WEIGHTS["synteza"] * sim_synteza
    )

    rows = []
    for i, code in enumerate(codes_ordered):
        # `prefix` pozwala zawęzić kandydatów do kodów zaczynających się od
        # już zatwierdzonych cyfr (np. moduł "1 cyfra przyporządkowana" -
        # pokazujemy tylko kody 4-cyfrowe pasujące do potwierdzonej 1. cyfry).
        if prefix and not code.startswith(prefix):
            continue
        rows.append(
            {
                "isco_code": code,
                "title_pl": metadata[code]["title"],
                "title_en": metadata[code].get("title_en", ""),
                "sim_title": round(float(sim_title[i]), 4),
                "sim_tasks": round(float(sim_tasks[i]), 4),
                "sim_synteza": round(float(sim_synteza[i]), 4),
                "score": round(float(score[i]), 4),
            }
        )

    columns = ["isco_code", "title_pl", "title_en", "sim_title", "sim_tasks", "sim_synteza", "score"]
    if not rows:
        # Żaden kod nie pasuje do podanego prefiksu - pd.DataFrame([]) nie
        # miałoby kolumny "score", więc sort_values("score") rzuciłby
        # KeyError. Zwracamy pusty DataFrame z właściwymi kolumnami.
        return pd.DataFrame(columns=columns)

    ranking = pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).reset_index(drop=True)
    return ranking.head(top_k)


def classify_level(
    zawod_czlowieka: str,
    umiejetnosci_obowiazki: str,
    model,
    title_emb: np.ndarray,
    tasks_emb: np.ndarray,
    synteza_emb: np.ndarray,
    codes_ordered: list,
    metadata: dict,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Wersja klasyfikacji na potrzeby trybu kaskadowego (kodowanie cyfra po cyfrze).

    Liczy podobieństwo do WSZYSTKICH kodów danego poziomu, opcjonalnie zawężonych
    do tych zaczynających się od `prefix` (czyli już wybranych wcześniej cyfr).
    Nie ucina wyniku do top_k - przy max 10 kandydatach na krok (kolejna cyfra 0-9)
    pokazujemy zawsze całą dostępną listę.
    """
    q_zawod = model.encode([QUERY_INSTRUCTION + zawod_czlowieka], normalize_embeddings=True)
    q_skills = model.encode([QUERY_INSTRUCTION + umiejetnosci_obowiazki], normalize_embeddings=True)

    sim_title = (title_emb @ q_zawod.T).flatten()
    sim_tasks = (tasks_emb @ q_skills.T).flatten()
    sim_synteza = (synteza_emb @ q_skills.T).flatten()

    score = (
        WEIGHTS["title"] * sim_title
        + WEIGHTS["tasks"] * sim_tasks
        + WEIGHTS["synteza"] * sim_synteza
    )

    rows = []
    for i, code in enumerate(codes_ordered):
        if prefix and not code.startswith(prefix):
            continue
        rows.append(
            {
                "isco_code": code,
                "title_pl": metadata[code]["title"],
                "title_en": metadata[code].get("title_en", ""),
                "sim_title": round(float(sim_title[i]), 4),
                "sim_tasks": round(float(sim_tasks[i]), 4),
                "sim_synteza": round(float(sim_synteza[i]), 4),
                "score": round(float(score[i]), 4),
            }
        )

    columns = ["isco_code", "title_pl", "title_en", "sim_title", "sim_tasks", "sim_synteza", "score"]
    if not rows:
        # Żaden kod na tym poziomie nie zaczyna się od dotychczas wybranego
        # prefiksu (ślepy zaułek w kaskadzie) - pd.DataFrame([]) nie miałoby
        # kolumny "score", więc sort_values("score") rzucałby KeyError.
        # Zwracamy pusty DataFrame z właściwymi kolumnami - render_cascade_step
        # obsługuje już ten przypadek (komunikat "Brak kodów ISCO-08
        # pasujących do wybranego dotychczas prefiksu...").
        return pd.DataFrame(columns=columns)

    ranking = pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).reset_index(drop=True)
    return ranking


def batch_classify_dataframe(
    zawody: list,
    obowiazki: list,
    model,
    title_emb: np.ndarray,
    tasks_emb: np.ndarray,
    synteza_emb: np.ndarray,
    codes_ordered: list,
    metadata: dict,
    top_k: int = 5,
) -> list:
    """
    Klasyfikuje całą listę zawodów na raz (wektorowo, bez pętli po modelu).
    Zwraca listę list słowników (top_k kandydatów dla każdego wiersza).
    """
    zawody = [str(z) if pd.notna(z) else "" for z in zawody]
    obowiazki = [str(o) if pd.notna(o) else "" for o in obowiazki]

    q_zawod = model.encode(
        [QUERY_INSTRUCTION + z for z in zawody], normalize_embeddings=True, show_progress_bar=False
    )
    q_skills = model.encode(
        [QUERY_INSTRUCTION + o for o in obowiazki], normalize_embeddings=True, show_progress_bar=False
    )

    sim_title = q_zawod @ title_emb.T        # (N, 436)
    sim_tasks = q_skills @ tasks_emb.T       # (N, 436)
    sim_synteza = q_skills @ synteza_emb.T   # (N, 436)

    score = (
        WEIGHTS["title"] * sim_title
        + WEIGHTS["tasks"] * sim_tasks
        + WEIGHTS["synteza"] * sim_synteza
    )  # (N, 436)

    results = []
    for row_idx in range(score.shape[0]):
        row_scores = score[row_idx]
        top_idx = np.argsort(row_scores)[::-1][:top_k]
        candidates = [
            {
                "isco_code": codes_ordered[i],
                "title_pl": metadata[codes_ordered[i]]["title"],
                "title_en": metadata[codes_ordered[i]].get("title_en", ""),
                "score": round(float(row_scores[i]), 4),
            }
            for i in top_idx
        ]
        results.append(candidates)

    return results


def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """Wczytuje CSV z autodetekcją separatora i kodowania."""
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)  # ostatnia próba, domyślne ustawienia


# ============================================================
# STAN SESJI - nawigacja między "stronami"
# ============================================================
if "page" not in st.session_state:
    # Po odświeżeniu strony (F5) odtwarzamy ostatnio otwartą "kartę" (metodę)
    # z parametru URL zamiast zawsze wracać do menu głównego (patrz go_to).
    st.session_state.page = st.query_params.get("p", "home")


def go_to(page_name: str):
    st.session_state.page = page_name
    st.query_params["p"] = page_name


def _questionnaire_csv() -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["kod_uczestnika", st.session_state.get("questionnaire_participant_code", "")])
    writer.writerow(["data", st.session_state.get("questionnaire_date", "")])
    writer.writerow(["wiek", st.session_state.get("questionnaire_age", "")])
    writer.writerow(["plec", st.session_state.get("questionnaire_gender", "")])
    writer.writerow([])
    writer.writerow(["czesc", "numer", "pytanie", "odpowiedz", "etykieta"])
    for section in QUESTIONNAIRE_SECTIONS:
        for number, question in enumerate(section["questions"], 1):
            answer = st.session_state.get(f"questionnaire_{section['id']}_{number}")
            writer.writerow([section["id"], number, question, answer, section["options"].get(answer, "")])
    return output.getvalue().encode("utf-8-sig")


def _questionnaire_answers() -> dict:
    return {
        f"{section['id']}{number}": st.session_state.get(
            f"questionnaire_{section['id']}_{number}"
        )
        for section in QUESTIONNAIRE_SECTIONS
        for number in range(1, len(section["questions"]) + 1)
    }


def _questionnaire_db() -> sqlite3.Connection:
    QUESTIONNAIRE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(QUESTIONNAIRE_DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS questionnaire_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            submitted_by TEXT NOT NULL,
            participant_code TEXT NOT NULL,
            survey_date TEXT,
            age INTEGER,
            gender TEXT,
            answers_json TEXT NOT NULL
        )
        """
    )
    return connection


def _save_questionnaire_response() -> int:
    now = _now_pl()
    survey_date = st.session_state.get("questionnaire_date")
    values = (
        now,
        st.session_state.get("username", ""),
        st.session_state.get("questionnaire_participant_code", "").strip(),
        survey_date.isoformat() if survey_date else None,
        st.session_state.get("questionnaire_age"),
        st.session_state.get("questionnaire_gender", "").strip(),
        json.dumps(_questionnaire_answers(), ensure_ascii=False),
    )
    response_id = st.session_state.get("questionnaire_response_id")
    with closing(_questionnaire_db()) as connection, connection:
        if response_id is None:
            cursor = connection.execute(
                """
                INSERT INTO questionnaire_responses
                    (submitted_at, updated_at, submitted_by, participant_code,
                     survey_date, age, gender, answers_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now,) + values,
            )
            response_id = int(cursor.lastrowid)
        else:
            connection.execute(
                """
                UPDATE questionnaire_responses
                SET updated_at = ?, submitted_by = ?, participant_code = ?,
                    survey_date = ?, age = ?, gender = ?, answers_json = ?
                WHERE id = ?
                """,
                values + (response_id,),
            )
    st.session_state.questionnaire_response_id = response_id
    return response_id


def _questionnaire_results_df() -> pd.DataFrame:
    with closing(_questionnaire_db()) as connection:
        rows = connection.execute(
            """
            SELECT id, submitted_at, updated_at, submitted_by, participant_code,
                   survey_date, age, gender, answers_json
            FROM questionnaire_responses
            ORDER BY id DESC
            """
        ).fetchall()
    columns = [
        "id", "submitted_at", "updated_at", "submitted_by", "participant_code",
        "survey_date", "age", "gender", "answers_json",
    ]
    records = []
    for row in rows:
        record = dict(zip(columns, row))
        answers = json.loads(record.pop("answers_json"))
        record.update(answers)
        records.append(record)
    answer_columns = [
        f"{section['id']}{number}"
        for section in QUESTIONNAIRE_SECTIONS
        for number in range(1, len(section["questions"]) + 1)
    ]
    return pd.DataFrame(
        records,
        columns=columns[:-1] + answer_columns,
    )


def _set_questionnaire_answer(key: str, value: int) -> None:
    st.session_state[key] = value


def render_questionnaire():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()
    if st.button("← Wróć do strony głównej", key="back_questionnaire"):
        go_to("home")
        st.rerun()

    st.title("Kwestionariusz badawczy")
    st.caption("Pakiet pytań dotyczących sposobu myślenia, działania i opisu siebie")
    st.info(
        "Nie ma odpowiedzi dobrych ani złych. Odpowiadaj samodzielnie i szczerze. "
        "Przy każdym pytaniu wybierz dokładnie jedną odpowiedź; możesz ją później zmienić."
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Kod uczestnika", key="questionnaire_participant_code")
            st.number_input("Wiek", min_value=0, max_value=120, value=None, step=1, key="questionnaire_age")
        with col2:
            st.date_input("Data", value=None, key="questionnaire_date")
            st.text_input("Płeć", key="questionnaire_gender")

    # Usuń wartości niepasujące do aktualnych skal (np. stare 0 w części E).
    for section in QUESTIONNAIRE_SECTIONS:
        for number in range(1, len(section["questions"]) + 1):
            key = f"questionnaire_{section['id']}_{number}"
            if key in st.session_state and st.session_state[key] not in section["options"]:
                del st.session_state[key]

    total = sum(len(section["questions"]) for section in QUESTIONNAIRE_SECTIONS)
    answered = sum(
        st.session_state.get(f"questionnaire_{section['id']}_{number}") is not None
        for section in QUESTIONNAIRE_SECTIONS
        for number in range(1, len(section["questions"]) + 1)
    )
    st.progress(answered / total, text=f"Udzielono odpowiedzi: {answered} z {total}")

    for section in QUESTIONNAIRE_SECTIONS:
        st.header(f"Część {section['id']}")
        st.subheader(section["title"])
        st.write(section["instruction"])
        with st.container(border=True, key=f"questionnaire_table_{section['id']}"):
            option_values = list(section["options"])
            header_columns = st.columns([5] + [1] * len(option_values), gap="small")
            with header_columns[0]:
                st.markdown(
                    '<div class="questionnaire-table-head questionnaire-table-head--question">Pytanie</div>',
                    unsafe_allow_html=True,
                )
            for column, value in zip(header_columns[1:], option_values):
                with column:
                    option_label = section["options"][value]
                    description = "" if option_label.strip() == str(value) else option_label
                    st.markdown(
                        '<div class="questionnaire-table-head">'
                        f'<span class="questionnaire-table-head__number">{value}</span>'
                        f'{description}</div>',
                        unsafe_allow_html=True,
                    )

            for number, question in enumerate(section["questions"], 1):
                answer_key = f"questionnaire_{section['id']}_{number}"
                row_columns = st.columns(
                    [5] + [1] * len(option_values), vertical_alignment="center", gap="small"
                )
                with row_columns[0]:
                    st.markdown(f"**{number}.** {question}")
                selected_value = st.session_state.get(answer_key)
                for column, value in zip(row_columns[1:], option_values):
                    with column:
                        st.button(
                            "●" if selected_value == value else "○",
                            key=f"questionnaire_choice_{section['id']}_{number}_{value}",
                            help=f"Wybierz odpowiedź {value}: {section['options'][value]}",
                            on_click=_set_questionnaire_answer,
                            args=(answer_key, value),
                            use_container_width=True,
                        )
                st.divider()
        st.write("")

    st.caption("Przed wysłaniem możesz wrócić do dowolnej części i zmienić każdą odpowiedź.")
    if st.button("Wyślij kwestionariusz", type="primary", use_container_width=True):
        missing = []
        for section in QUESTIONNAIRE_SECTIONS:
            for number in range(1, len(section["questions"]) + 1):
                if st.session_state.get(f"questionnaire_{section['id']}_{number}") is None:
                    missing.append(f"{section['id']}{number}")
        if not st.session_state.get("questionnaire_participant_code", "").strip():
            st.error("Wpisz kod uczestnika.")
        elif missing:
            st.error(f"Odpowiedz na wszystkie pytania. Brakujące pozycje: {', '.join(missing)}.")
        else:
            try:
                response_id = _save_questionnaire_response()
            except sqlite3.Error:
                st.error("Nie udało się zapisać odpowiedzi. Spróbuj ponownie lub skontaktuj się z administratorem.")
            else:
                st.session_state.questionnaire_completed = True
                st.success(
                    f"Odpowiedzi zapisano trwale (rekord nr {response_id}). "
                    "Nadal możesz je zmienić i wysłać ponownie."
                )

    if st.session_state.get("questionnaire_completed"):
        st.download_button(
            "Pobierz odpowiedzi (CSV)", data=_questionnaire_csv(),
            file_name="odpowiedzi_kwestionariusz.csv", mime="text/csv", use_container_width=True,
        )


def render_questionnaire_results():
    if st.session_state.get("username") not in ADMIN_USERS:
        st.error("Brak uprawnień do wyników ankiet.")
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()
    if st.button("← Wróć do strony głównej", key="back_questionnaire_results"):
        go_to("home")
        st.rerun()

    st.title("Wyniki ankiet")
    results = _questionnaire_results_df()
    st.metric("Liczba zapisanych ankiet", len(results))
    if results.empty:
        st.info("Nie zapisano jeszcze żadnej ankiety.")
        return

    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Pobierz wszystkie wyniki (CSV)",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="wszystkie_wyniki_ankiet.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# EWIDENCJA POSTĘPU KODOWANIA (przydział z Status.csv + zapisy na bieżąco)
# ============================================================
WARIANT_LETTER_TO_LABEL = {"A": "Ręczne", "B": "AI", "C": "AI (1 cyfra)"}

# Normalizacja aliasów kodera używanych w pliku Status.csv (różne w Lipcu
# i Sierpniu) do jednolitej postaci koder_1 / koder_2 / koder_3, niezależnie
# od miesiąca - tak, żeby w ewidencji nie pojawiały się imiona.
KODER_ALIAS_TO_NORMALIZED = {
    "Jan": "koder_1",
    "Marta": "koder_2",
    "Piotr": "koder_3",
    "Koder_1": "koder_1",
    "Koder_2": "koder_2",
    "Koder_3": "koder_3",
}

_PACZKA_RE = re.compile(r"^(?P<miesiac>[^_]+)_(?P<koder>.+)_(?P<wariant>[ABC])$")

# Pełna lista kolumn "wynikowych" (kody, uzasadnienia, czasy, oceny AI) zbieranych
# do szczegółowego logu kodowania (coding_details) - patrz _record_coding_details.
# Świadomie NIE obejmuje surowych zmiennych ankietowych (B31, B33...) respondenta.
CODING_DETAIL_COLUMNS = [
    "ISCO_wybrany", "ISCO_PRED",
    "ISCO_poziom1", "ISCO_poziom2", "ISCO_poziom3", "ISCO_poziom4",
    "ISCO_poziom1_zmienne", "ISCO_poziom2_zmienne", "ISCO_poziom3_zmienne", "ISCO_poziom4_zmienne",
    "ISCO_poziom1_ranking_pozycja", "ISCO_poziom2_ranking_pozycja",
    "ISCO_poziom3_ranking_pozycja", "ISCO_poziom4_ranking_pozycja",
    "ISCO_poziom1_score", "ISCO_poziom2_score", "ISCO_poziom3_score", "ISCO_poziom4_score",
    "Decyzja_kodera_zawod", "Decyzja_kodera_notatka",
    "Brak_mozliwosci_zakodowania", "Uzasadnienie_finalne",
    "Ranking_pozycja_wybranego_kodu", "Score_wybranego_kodu",
    "Ocena_AI_top10_1_5", "Ocena_AI_kaskadowo_1_5",
    "Cyfra1_zatwierdzona_expert", "Powod_odrzucenia_cyfry",
    "Czas_kodowania_sekundy", "Czas_do_pierwszej_interakcji_sekundy", "Czy_uzytkownik_wracal",
]


def _parse_paczka(paczka: str) -> Optional[dict]:
    """Rozbija wartość kolumny 'paczka' z pliku Status.csv na miesiąc, znormalizowanego
    kodera (koder_1/2/3, niezależnie od tego czy w pliku były imiona czy 'Koder_N')
    i etykietę wariantu (Ręczne/AI/AI (1 cyfra)). Zwraca None dla 'Trening' (osobna
    pula kalibracyjna, bez przypisanego 1:1 kodera/wariantu w tym pliku) i dla
    nierozpoznanego formatu. 'Brak' jest obsługiwane osobno przez wywołującego
    (patrz _import_case_assignments) - trafia do case_unassigned, nie tutaj."""
    paczka = (paczka or "").strip()
    if paczka in ("Brak", "Trening", ""):
        return None
    match = _PACZKA_RE.match(paczka)
    if not match:
        return None
    miesiac = match.group("miesiac")
    koder_raw = match.group("koder")
    wariant_letter = match.group("wariant")
    return {
        "miesiac": miesiac,
        "koder_przydzielony": KODER_ALIAS_TO_NORMALIZED.get(koder_raw, koder_raw),
        "wariant": WARIANT_LETTER_TO_LABEL.get(wariant_letter, wariant_letter),
    }


def _coding_db() -> sqlite3.Connection:
    CODING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CODING_DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    # Przydział (kto ma zakodować co, jakim wariantem) - wgrywany RAZ przez admina
    # i trwale przechowywany na serwerze (patrz render_ewidencja) - kolejne wejścia
    # na stronę NIE wymagają ponownego wgrywania, chyba że admin świadomie podmieni plik.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS case_assignments (
            idno TEXT NOT NULL,
            osoba TEXT NOT NULL,
            miesiac TEXT NOT NULL,
            koder_przydzielony TEXT NOT NULL,
            wariant TEXT NOT NULL,
            PRIMARY KEY (idno, osoba, wariant)
        )
        """
    )
    # Przypadki oznaczone w Status.csv jako 'Brak' (nieprzydzielone do nikogo) -
    # trzymane osobno, żeby nie zaśmiecały głównej ewidencji (patrz druga zakładka).
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS case_unassigned (
            idno TEXT NOT NULL,
            osoba TEXT NOT NULL,
            PRIMARY KEY (idno, osoba)
        )
        """
    )
    # Metadane ostatniego importu przydziału - jeden wiersz (id=1), pokazywany w UI,
    # żeby było jasne, że plik jest wgrywany raz, a nie za każdym razem od nowa.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS case_assignments_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            imported_at TEXT NOT NULL,
            imported_by TEXT NOT NULL,
            assigned_rows INTEGER NOT NULL,
            unassigned_rows INTEGER NOT NULL
        )
        """
    )
    # Status "zrobione/nie" per (idno, osoba, wariant) - nadpisywany przy ponownym
    # zakodowaniu tego samego przypadku (patrz _record_coding_event).
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS coding_events (
            idno TEXT NOT NULL,
            osoba TEXT NOT NULL,
            wariant TEXT NOT NULL,
            koder TEXT NOT NULL,
            isco_kod TEXT,
            zapisano_o TEXT NOT NULL,
            PRIMARY KEY (idno, osoba, wariant)
        )
        """
    )
    # Pełny, dopisywany (append-only) log przebiegu kodowania - osobna tabela,
    # NIE łączona z bazową ramką ewidencji (coding_events / _ewidencja_df).
    # Każdy zapis decyzji dodaje nowy wiersz (historia, nie nadpisywanie), żeby
    # było widać też ewentualne poprawki/ponowne kodowanie tego samego przypadku.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS coding_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idno TEXT NOT NULL,
            osoba TEXT NOT NULL,
            wariant TEXT NOT NULL,
            koder TEXT NOT NULL,
            zapisano_o TEXT NOT NULL,
            szczegoly_json TEXT NOT NULL
        )
        """
    )
    return connection


def _normalize_idno(idno) -> str:
    idno = str(idno).strip()
    try:
        idno = str(int(float(idno)))
    except (ValueError, TypeError):
        pass
    return idno


def _import_case_assignments(status_df: pd.DataFrame, imported_by: str) -> Tuple[int, int, int]:
    """Wczytuje plik Status.csv (kolumny: idno, osoba, paczka) i CAŁKOWICIE
    zastępuje poprzedni przydział - zarówno przypisane przypadki (case_assignments),
    jak i nieprzydzielone (case_unassigned, dawne 'Brak'). Wiersze 'Trening' oraz
    o nierozpoznanym formacie paczki są pomijane. Zwraca
    (liczba_przypisanych, liczba_nieprzydzielonych, liczba_pominiętych)."""
    assigned_rows = []
    unassigned_rows = []
    skipped = 0
    for _, r in status_df.iterrows():
        idno = _normalize_idno(r.get("idno", ""))
        osoba = str(r.get("osoba", "")).strip()
        paczka_raw = str(r.get("paczka", "")).strip()
        if not idno or not osoba:
            skipped += 1
            continue
        if paczka_raw == "Brak":
            unassigned_rows.append((idno, osoba))
            continue
        parsed = _parse_paczka(paczka_raw)
        if parsed is None:
            skipped += 1
            continue
        assigned_rows.append((idno, osoba, parsed["miesiac"], parsed["koder_przydzielony"], parsed["wariant"]))

    now = _now_pl()
    with closing(_coding_db()) as connection, connection:
        connection.execute("DELETE FROM case_assignments")
        connection.execute("DELETE FROM case_unassigned")
        connection.executemany(
            """
            INSERT OR REPLACE INTO case_assignments (idno, osoba, miesiac, koder_przydzielony, wariant)
            VALUES (?, ?, ?, ?, ?)
            """,
            assigned_rows,
        )
        connection.executemany(
            "INSERT OR REPLACE INTO case_unassigned (idno, osoba) VALUES (?, ?)",
            unassigned_rows,
        )
        connection.execute(
            """
            INSERT INTO case_assignments_meta (id, imported_at, imported_by, assigned_rows, unassigned_rows)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                imported_at = excluded.imported_at,
                imported_by = excluded.imported_by,
                assigned_rows = excluded.assigned_rows,
                unassigned_rows = excluded.unassigned_rows
            """,
            (now, imported_by, len(assigned_rows), len(unassigned_rows)),
        )
    return len(assigned_rows), len(unassigned_rows), skipped


def _assignment_meta() -> Optional[dict]:
    with closing(_coding_db()) as connection:
        row = connection.execute(
            "SELECT imported_at, imported_by, assigned_rows, unassigned_rows FROM case_assignments_meta WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    return {
        "imported_at": row[0], "imported_by": row[1],
        "assigned_rows": row[2], "unassigned_rows": row[3],
    }


def _record_coding_event(idno, osoba: str, wariant: str, koder: str, isco_kod: Optional[str]) -> None:
    """Zapisuje na bieżąco (real-time) na serwerze fakt zakodowania danego
    przypadku - wywoływane z _save_respondent_meta przy każdym zapisie decyzji,
    niezależnie od modułu (A/B/C). Nadpisuje poprzedni wpis, jeśli ten sam
    przypadek zostanie zakodowany ponownie (np. poprawka) - to jest lekka
    "bazowa ramka" statusu, patrz też _record_coding_details dla pełnego logu."""
    idno = _normalize_idno(idno)
    now = _now_pl()
    with closing(_coding_db()) as connection, connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO coding_events (idno, osoba, wariant, koder, isco_kod, zapisano_o)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (idno, osoba, wariant, koder, isco_kod, now),
        )


def _record_coding_details(idno, osoba: str, wariant: str, koder: str, row: pd.Series) -> None:
    """Dopisuje (append, nie nadpisuje) pełny zestaw szczegółów przebiegu
    kodowania - kody na każdym poziomie, użyte zmienne, czasy, oceny AI,
    uzasadnienia - do OSOBNEJ tabeli coding_details, celowo poza bazową ramką
    ewidencji statusu (coding_events / _ewidencja_df)."""
    idno = _normalize_idno(idno)
    szczegoly = {}
    for col in CODING_DETAIL_COLUMNS:
        if col in row.index:
            val = row[col]
            szczegoly[col] = None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)
    now = _now_pl()
    with closing(_coding_db()) as connection, connection:
        connection.execute(
            """
            INSERT INTO coding_details (idno, osoba, wariant, koder, zapisano_o, szczegoly_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (idno, osoba, wariant, koder, now, json.dumps(szczegoly, ensure_ascii=False, default=str)),
        )


def _ewidencja_df() -> pd.DataFrame:
    """Łączy przydział przypisanych przypadków (case_assignments) z faktycznymi
    zapisami postępu (coding_events) po kluczu (idno, osoba, wariant). Nie
    obejmuje 'Brak' (patrz _unassigned_df, osobna zakładka)."""
    with closing(_coding_db()) as connection:
        assignments = pd.read_sql_query("SELECT * FROM case_assignments", connection)
        events = pd.read_sql_query("SELECT * FROM coding_events", connection)

    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "idno", "osoba", "miesiac", "koder_przydzielony", "wariant",
                "status", "koder_faktyczny", "zapisano_o", "isco_kod",
            ]
        )

    merged = assignments.merge(events, on=["idno", "osoba", "wariant"], how="left")
    merged["status"] = np.where(merged["koder"].notna(), "Wykonane", "Niewykonane")
    merged = merged.rename(columns={"koder": "koder_faktyczny"})
    return merged[
        [
            "idno", "osoba", "miesiac", "koder_przydzielony", "wariant",
            "status", "koder_faktyczny", "zapisano_o", "isco_kod",
        ]
    ].sort_values(["miesiac", "koder_przydzielony", "wariant", "idno"])


def _unassigned_df() -> pd.DataFrame:
    with closing(_coding_db()) as connection:
        return pd.read_sql_query(
            "SELECT idno, osoba FROM case_unassigned ORDER BY idno", connection
        )


def _coding_details_df() -> pd.DataFrame:
    with closing(_coding_db()) as connection:
        raw = pd.read_sql_query(
            "SELECT id, idno, osoba, wariant, koder, zapisano_o, szczegoly_json "
            "FROM coding_details ORDER BY id DESC",
            connection,
        )
    if raw.empty:
        return raw
    details = pd.json_normalize(raw["szczegoly_json"].apply(json.loads))
    return pd.concat([raw.drop(columns=["szczegoly_json"]), details], axis=1)


def _case_lookup(idno: str, osoba: str) -> dict:
    """Zwraca dla danego (idno, osoba): status każdego przydzielonego wariantu
    (wykonane/niewykonane, kto i kiedy), listę wariantów, których jeszcze
    brakuje, oraz informację czy przypadek w ogóle jest przydzielony (czy
    może jest w puli 'Brak'). Używane w zakładce 'Sprawdź case'."""
    idno_norm = _normalize_idno(idno)
    with closing(_coding_db()) as connection:
        assignments = pd.read_sql_query(
            "SELECT * FROM case_assignments WHERE idno = ? AND osoba = ?",
            connection, params=(idno_norm, osoba),
        )
        events = pd.read_sql_query(
            "SELECT * FROM coding_events WHERE idno = ? AND osoba = ?",
            connection, params=(idno_norm, osoba),
        )
        is_brak = pd.read_sql_query(
            "SELECT 1 FROM case_unassigned WHERE idno = ? AND osoba = ?",
            connection, params=(idno_norm, osoba),
        ).shape[0] > 0

    if assignments.empty:
        return {"found": False, "is_brak": is_brak, "table": pd.DataFrame(), "brakujace": []}

    merged = assignments.merge(events, on=["idno", "osoba", "wariant"], how="left")
    merged["status"] = np.where(merged["koder"].notna(), "Wykonane", "Niewykonane")
    merged = merged.rename(columns={"koder": "koder_faktyczny"})
    brakujace = merged.loc[merged["status"] == "Niewykonane", "wariant"].tolist()
    table = merged[
        ["wariant", "koder_przydzielony", "status", "koder_faktyczny", "zapisano_o", "isco_kod"]
    ].sort_values("wariant")
    return {"found": True, "is_brak": False, "table": table, "brakujace": brakujace}


def render_ewidencja():
    if st.session_state.get("username") not in ADMIN_USERS:
        st.error("Brak uprawnień do ewidencji postępu kodowania.")
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()
    if st.button("← Wróć do strony głównej", key="back_ewidencja"):
        go_to("home")
        st.rerun()

    st.title("Ewidencja postępu kodowania")

    meta = _assignment_meta()
    with st.expander("Podmień przydział (Status.csv)", expanded=meta is None):
        if meta is not None:
            st.caption(
                f"Aktualnie wgrany przydział: {meta['assigned_rows']} przypisanych + "
                f"{meta['unassigned_rows']} nieprzydzielonych ('Brak'), zaimportowany "
                f"{meta['imported_at']} przez {meta['imported_by']}."
            )
        else:
            st.caption("Nie wgrano jeszcze żadnego przydziału.")
        st.caption("Wgranie pliku CAŁKOWICIE zastępuje poprzedni przydział.")
        # Klucz widgetu zawiera licznik, który zwiększamy po każdym udanym imporcie
        # (patrz niżej) - wymusza to całkowicie NOWY widget file_uploader przy
        # kolejnym wejściu, zamiast pozostawiać poprzednio wybrany plik "przyklejony"
        # do starego klucza (przez co podmiana na inny plik czasem nie działała).
        uploader_key = f"uploader_status_{st.session_state.get('uploader_status_generation', 0)}"
        status_file = st.file_uploader("Wybierz plik CSV", type=["csv"], key=uploader_key)
        if status_file is not None and st.button("Importuj / podmień przydział", key="import_status_btn"):
            status_df = read_csv_robust(status_file)
            assigned, unassigned, skipped = _import_case_assignments(
                status_df, imported_by=st.session_state.get("username", "")
            )
            st.session_state["uploader_status_generation"] = (
                st.session_state.get("uploader_status_generation", 0) + 1
            )
            st.success(
                f"Zaimportowano {assigned} przypisanych i {unassigned} nieprzydzielonych "
                f"('Brak'), pominięto {skipped} wierszy (Trening/nierozpoznane)."
            )
            st.rerun()

    tab_ewidencja, tab_brak, tab_lookup, tab_log = st.tabs(
        ["Ewidencja", "Nieprzydzielone (Brak)", "Sprawdź case", "Log szczegółowy"]
    )

    with tab_ewidencja:
        df = _ewidencja_df()
        if df.empty:
            st.info("Brak zaimportowanego przydziału - wgraj plik Status.csv powyżej.")
        else:
            total = len(df)
            done = int((df["status"] == "Wykonane").sum())
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Przydzielonych przypadków", total)
            col_m2.metric("Wykonanych", done)
            col_m3.metric("Pozostało", total - done)

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                miesiac_sel = st.selectbox("Miesiąc", ["Wszystkie"] + sorted(df["miesiac"].unique().tolist()))
            with col_f2:
                koder_sel = st.selectbox(
                    "Koder przydzielony", ["Wszyscy"] + sorted(df["koder_przydzielony"].unique().tolist())
                )
            with col_f3:
                wariant_sel = st.selectbox("Wariant", ["Wszystkie"] + sorted(df["wariant"].unique().tolist()))
            with col_f4:
                status_sel = st.selectbox("Status", ["Wszystkie", "Wykonane", "Niewykonane"])

            filtered = df.copy()
            if miesiac_sel != "Wszystkie":
                filtered = filtered[filtered["miesiac"] == miesiac_sel]
            if koder_sel != "Wszyscy":
                filtered = filtered[filtered["koder_przydzielony"] == koder_sel]
            if wariant_sel != "Wszystkie":
                filtered = filtered[filtered["wariant"] == wariant_sel]
            if status_sel != "Wszystkie":
                filtered = filtered[filtered["status"] == status_sel]

            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "Pobierz widoczną tabelę (CSV)",
                data=filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name="ewidencja_kodowania.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_brak:
        brak_df = _unassigned_df()
        st.metric("Nieprzydzielonych ('Brak')", len(brak_df))
        if brak_df.empty:
            st.info("Brak nieprzydzielonych przypadków w zaimportowanym pliku.")
        else:
            st.dataframe(brak_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Pobierz listę nieprzydzielonych (CSV)",
                data=brak_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="nieprzydzielone_brak.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_lookup:
        st.caption("Sprawdź, którą metodą dany przypadek został (lub nie) zakodowany.")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            lookup_idno = st.text_input("IDNO", key="lookup_idno").strip()
        with col_l2:
            lookup_osoba = st.selectbox("Osoba", ["Respondent", "Partner"], key="lookup_osoba")

        if lookup_idno:
            result = _case_lookup(lookup_idno, lookup_osoba)
            if not result["found"]:
                if result["is_brak"]:
                    st.warning("Ten przypadek jest w puli 'Brak' - nieprzydzielony do żadnego kodera/wariantu.")
                else:
                    st.warning("Nie znaleziono takiego przypadku w zaimportowanym przydziale.")
            else:
                st.dataframe(result["table"], use_container_width=True, hide_index=True)
                if result["brakujace"]:
                    st.warning("Brakuje jeszcze: " + ", ".join(result["brakujace"]))
                else:
                    st.success("Wszystkie przydzielone warianty wykonane.")

    with tab_log:
        st.caption(
            "Pełny, chronologiczny log każdego zapisu decyzji (kody na każdym poziomie, "
            "użyte zmienne, czasy, oceny AI, uzasadnienia) - osobno od bazowej ewidencji "
            "statusu powyżej. Jeden przypadek może mieć kilka wpisów, jeśli był kodowany "
            "ponownie."
        )
        details_df = _coding_details_df()
        if details_df.empty:
            st.info("Brak zapisanych jeszcze żadnych szczegółów kodowania.")
        else:
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Pobierz pełny log kodowania (CSV)",
                data=details_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="log_szczegolowy_kodowania.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# STRONA GŁÓWNA
# ============================================================
def render_home():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()
    st.markdown(
        """
        <div class="app-header">
            <h1>System wspomagania klasyfikacji zawodów ISCO-08</h1>
            <p>na podstawie danych European Social Survey (ESS)</p>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Przejdź do kwestionariusza badawczego", type="primary", use_container_width=True):
        go_to("questionnaire")
        st.rerun()

    st.write(
        "Aplikacja umożliwia klasyfikację zawodów zgodnie ze standardem ISCO-08 "
        "z wykorzystaniem modeli sztucznej inteligencji, na podstawie danych ankietowych "
        "European Social Survey (ESS). System wspiera klasyfikację zawodów wspomaganą "
        "decyzją eksperta."
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Metoda A'
                '<span class="module-card-sub">Kodowanie ręczne</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_manual", use_container_width=True):
                go_to("classify_manual")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Metoda B'
                '<span class="module-card-sub">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_hitl", use_container_width=True):
                go_to("classify_hitl")
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Metoda C'
                '<span class="module-card-sub">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta'
                '<br>(1&nbsp;cyfra&nbsp;przyporządkowana)</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_hitl_1digit", use_container_width=True):
                go_to("classify_hitl_1digit")
                st.rerun()


# ============================================================
# STRONA: METODA A (KODOWANIE RĘCZNE)
# ============================================================
def _manual_level_options(level: int, prefix: str = "") -> list[dict]:
    _, _, _, codes_ordered, metadata = load_embeddings_level(level)
    codes = sorted(code for code in codes_ordered if not prefix or code.startswith(prefix))
    options = [
        {
            "code": code,
            "label": _format_candidate_label(
                code,
                metadata[code]["title"],
                metadata[code].get("title_en", ""),
                None,
            ),
        }
        for code in codes
    ]
    if level > 1:
        # Poziom 2-4: nie da się ustalić dokładniejszej cyfry, ale kierunek
        # (poprzednie cyfry) jest znany - dopełniamy resztę zerami i zapisujemy
        # jako wynik częściowej precyzji (analogicznie do trybu kaskadowego
        # w module B/C - patrz render_cascade_step / NO_DETERMINATION_OPTION).
        preview_code = (prefix + "0" * (5 - level))[:4]
        options.append({
            "code": "__NO_MATCH__",
            "label": f"Brak możliwości ustalenia dokładnej cyfry (dopełnij pozostałe cyfry zerami, kod: {preview_code})",
        })
    else:
        # Poziom 1: brak nawet grupy głównej - to prawdziwie niemożliwy do
        # zakodowania przypadek, nie zapisujemy żadnego kodu (patrz obsługa
        # "__UNCODABLE__" w render_manual_step) - tak samo jak "Brak możliwości
        # zakodowania..." w module B/C.
        options.append({
            "code": "__UNCODABLE__",
            "label": "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)",
        })
    return options


def _init_manual_result_columns(df: pd.DataFrame) -> None:
    if "ISCO_wybrany" not in df.columns:
        df["ISCO_wybrany"] = None
        df["Decyzja_kodera_zawod"] = None
        df["Decyzja_kodera_notatka"] = None
    if "Kodowany_podmiot" not in df.columns:
        df["Kodowany_podmiot"] = None
    if "Brak_mozliwosci_zakodowania" not in df.columns:
        df["Brak_mozliwosci_zakodowania"] = None
    for col in ("ISCO_poziom1", "ISCO_poziom2", "ISCO_poziom3", "ISCO_poziom4", "ISCO_PRED"):
        if col not in df.columns:
            df[col] = None
    for col in (
        "Uzasadnienie_finalne",
        "Czas_kodowania_sekundy",
        "Czas_do_pierwszej_interakcji_sekundy",
        "Czy_uzytkownik_wracal",
    ):
        if col not in df.columns:
            df[col] = None
    for col in (
        "ISCO_wybrany",
        "Decyzja_kodera_zawod",
        "Decyzja_kodera_notatka",
        "Kodowany_podmiot",
        "Brak_mozliwosci_zakodowania",
        "ISCO_poziom1",
        "ISCO_poziom2",
        "ISCO_poziom3",
        "ISCO_poziom4",
        "ISCO_PRED",
        "Uzasadnienie_finalne",
    ):
        _ensure_text_column_dtype(df, col)
    _ensure_object_dtype(df, "Czy_uzytkownik_wracal")


def _manual_reset_idx(idx: int) -> None:
    st.session_state[f"manual_step_{idx}"] = 1
    st.session_state[f"manual_digits_{idx}"] = []


def _manual_save_uncodable(
    df,
    idx: int,
    target: str,
    uzasadnienie: str,
    df_state_key: str,
    idx_state_key: str,
    qualifying_positions: list[int],
) -> None:
    """Zapisuje przypadek jako NIEMOŻLIWY do zakodowania (poziom 1 - brak
    nawet grupy głównej) - analogicznie do NO_CODE_OPTION w module B/C: nie
    zapisuje żadnego kodu ISCO, tylko ustawia Brak_mozliwosci_zakodowania i
    przechodzi do kolejnego przypadku."""
    df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
    df.at[idx, "Kodowany_podmiot"] = target
    if uzasadnienie.strip():
        df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie.strip()
        df.at[idx, "Decyzja_kodera_notatka"] = uzasadnienie.strip()
    else:
        df.at[idx, "Uzasadnienie_finalne"] = None
        df.at[idx, "Decyzja_kodera_notatka"] = None
    _save_respondent_meta(df, idx, df_state_key=df_state_key)
    _persist_df(df_state_key, df)
    st.session_state.pop(f"manual_step_{idx}", None)
    st.session_state.pop(f"manual_digits_{idx}", None)
    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, len(df)))


def _manual_save_code(
    df,
    idx: int,
    final_code: str,
    target: str,
    uzasadnienie: str,
    df_state_key: str,
    idx_state_key: str,
    qualifying_positions: list[int],
    selected_vars: Optional[list[str]] = None,
) -> None:
    """Zapisuje kompletną decyzję Metody A i przechodzi do następnego przypadku."""
    for level, digit in enumerate(final_code, start=1):
        df.at[idx, f"ISCO_poziom{level}"] = digit
    current_level = len(digits_so_far := st.session_state.get(f"manual_digits_{idx}", [])) + 1
    df.at[idx, f"ISCO_poziom{current_level}_zmienne"] = ", ".join(selected_vars) if selected_vars else None
    df.at[idx, "ISCO_PRED"] = final_code
    df.at[idx, "ISCO_wybrany"] = final_code
    df.at[idx, "Kodowany_podmiot"] = target
    df.at[idx, "Brak_mozliwosci_zakodowania"] = None
    if uzasadnienie.strip():
        df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie.strip()
        df.at[idx, "Decyzja_kodera_notatka"] = uzasadnienie.strip()
    else:
        # Przy ponownym kodowaniu nie pozostawiamy komentarza ze starej decyzji.
        df.at[idx, "Uzasadnienie_finalne"] = None
        df.at[idx, "Decyzja_kodera_notatka"] = None
    _save_respondent_meta(df, idx, df_state_key=df_state_key)
    _persist_df(df_state_key, df)
    st.session_state.pop(f"manual_step_{idx}", None)
    st.session_state.pop(f"manual_digits_{idx}", None)
    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, len(df)))


def _ctrl_enter_shortcut(
    handler_key: str,
    fallback_labels: list[str],
    direct_input_aria_label: Optional[str] = None,
    direct_save_label: Optional[str] = None,
) -> None:
    """Uniwersalny skrót klawiszowy Ctrl+Enter (na macOS równiez Cmd+Enter,
    dzięki nasłuchiwaniu jednocześnie na ctrlKey i metaKey - stąd działa tak
    samo na Windows, Linux i macOS) uruchamiający główny przycisk "dalej" na
    bieżącym ekranie kodowania.

    Jeśli podano `direct_input_aria_label` i `direct_save_label`, a pole o tej
    etykiecie ma aktualnie wpisaną wartość, klikany jest przycisk zapisu kodu
    wpisanego ręcznie (`direct_save_label`) zamiast przycisków z listy
    `fallback_labels` - analogicznie jak w Metodzie A i kaskadzie, gdzie oba
    mechanizmy (wybór z listy / wpisanie kodu) mają osobne przyciski."""
    fallback_json = json.dumps(fallback_labels)
    direct_save_json = json.dumps(direct_save_label) if direct_save_label else "null"
    aria_json = json.dumps(direct_input_aria_label) if direct_input_aria_label else "null"
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const handlerKey = '{handler_key}';
        if (window.parent[handlerKey]) {{
            doc.removeEventListener('keydown', window.parent[handlerKey], true);
        }}
        window.parent[handlerKey] = (event) => {{
            if (!(event.ctrlKey || event.metaKey) || event.key !== 'Enter') return;
            const ariaLabel = {aria_json};
            const directSaveLabel = {direct_save_json};
            const directInput = ariaLabel ? doc.querySelector(`input[aria-label="${{ariaLabel}}"]`) : null;
            const labels = (directInput && directInput.value.trim() && directSaveLabel)
                ? [directSaveLabel]
                : {fallback_json};
            const buttons = [...doc.querySelectorAll('button')];
            const button = buttons.find((item) => labels.includes(item.innerText.trim()));
            if (!button || button.disabled) return;
            event.preventDefault();
            const active = doc.activeElement;
            const clickButton = () => button.click();
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {{
                // Ctrl+Enter NIE przenosi fokusu poza pole (w przeciwieństwie do
                // kliknięcia myszą gdzie indziej), a Streamlit wysyła do serwera
                // właśnie wpisaną wartość pola tekstowego dopiero przy utracie
                // fokusu (blur) - bez tego serwer mógł jeszcze "widzieć" starą
                // wartość sprzed edycji (np. z poprzedniego przypadku) w
                // momencie kliknięcia przycisku. Wymuszamy więc blur i dajemy
                // Streamlitowi chwilę na przetworzenie synchronizacji, zanim
                // faktycznie klikniemy przycisk zapisu.
                active.blur();
                setTimeout(clickButton, 60);
            }} else {{
                clickButton();
            }}
        }};
        doc.addEventListener('keydown', window.parent[handlerKey], true);
        </script>
        """,
        height=0,
    )


def _instant_text_persistence(idx: int, fields: list[tuple[str, str]]) -> None:
    """Zapisuje treść pól tekstowych (textarea/input) o podanych aria-label do
    localStorage przeglądarki PRZY KAŻDYM naciśnięciu klawisza ('input'), nie
    czekając na utratę fokusu jak robi to domyślnie Streamlit (które
    synchronizuje wartość z serwerem dopiero po kliknięciu poza polem). Dzięki
    temu tekst przetrwa nawet natychmiastowe odświeżenie strony w trakcie
    pisania - inaczej niż przy zwykłej synchronizacji Streamlita, gdzie
    wpisany, ale jeszcze niewysłany tekst ginie bezpowrotnie przy F5.

    `fields` to lista par (aria_label, aktualna_wartość) - aktualna_wartość to
    to, co Streamlit WŁAŚNIE wyrenderował dla tego przypadku (pusty string dla
    nigdy niekodowanego przypadku, albo wcześniej zapisany kod/uzasadnienie
    przy powrocie do już zakodowanego - patrz _seed_direct_code_and_uzasadnienie
    / _seed_top10_widgets). Przy KAŻDEJ zmianie `idx` skrypt WYMUSZA na polu
    dokładnie tę wartość - chyba że w localStorage czeka świeższy, jeszcze
    niezapisany szkic dla TEGO SAMEGO idx (czyli odświeżenie strony w trakcie
    pisania). Dzięki jawnemu wymuszaniu wartości (a nie tylko jej
    'dokładaniu', gdy pasuje) pole nie może już zostać z resztką tekstu po
    poprzednim przypadku - to była przyczyna błędu, w którym ręcznie wpisany
    kod 'zostawał' w polu przy przejściu do kolejnego, nigdy niekodowanego
    przypadku."""
    labels_json = json.dumps([label for label, _ in fields])
    fields_json = json.dumps([{"label": label, "value": value} for label, value in fields])
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const idx = {idx};
            const labels = {labels_json};
            const fields = {fields_json};
            const STORAGE_KEY = 'isco_draft_v1';

            function loadStore() {{
                try {{ return JSON.parse(window.parent.localStorage.getItem(STORAGE_KEY) || '{{}}'); }}
                catch (e) {{ return {{}}; }}
            }}
            function saveStore(store) {{
                try {{ window.parent.localStorage.setItem(STORAGE_KEY, JSON.stringify(store)); }}
                catch (e) {{}}
            }}
            function nativeSet(el, value) {{
                const proto = el.tagName === 'TEXTAREA'
                    ? window.parent.HTMLTextAreaElement.prototype
                    : window.parent.HTMLInputElement.prototype;
                const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                setter.call(el, value);
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}

            if (!window.parent.__iscoInstantSaveInstalled) {{
                window.parent.__iscoInstantSaveInstalled = true;
                doc.addEventListener('input', (event) => {{
                    const el = event.target;
                    const label = el.getAttribute && el.getAttribute('aria-label');
                    if (!label || !window.parent.__iscoTrackedLabels || !window.parent.__iscoTrackedLabels.includes(label)) return;
                    const store = loadStore();
                    store[label] = {{ idx: window.parent.__iscoCurrentIdx, value: el.value }};
                    saveStore(store);
                }}, true);
            }}
            window.parent.__iscoTrackedLabels = labels;
            window.parent.__iscoCurrentIdx = idx;

            function enforceAll() {{
                const store = loadStore();
                fields.forEach(({{label, value}}) => {{
                    const el = doc.querySelector(`textarea[aria-label="${{label}}"], input[aria-label="${{label}}"]`);
                    if (!el) return;
                    // Raz na idx wystarczy - inaczej wymuszalibyśmy tę samą
                    // wartość przy każdej mutacji DOM, nadpisując na siłę to,
                    // co koder właśnie wpisuje.
                    if (el.dataset.iscoIdx === String(idx)) return;
                    el.dataset.iscoIdx = String(idx);
                    const draft = store[label];
                    if (draft && draft.idx === idx && draft.value && draft.value !== value) {{
                        // Niezapisany jeszcze szkic z tej samej sesji dla TEGO
                        // SAMEGO przypadku (np. po odświeżeniu F5 w trakcie
                        // pisania) - ma pierwszeństwo przed wartością z serwera.
                        nativeSet(el, draft.value);
                    }} else if (el.value !== value) {{
                        // W każdym innym wypadku wymuszamy DOKŁADNIE to, co
                        // Streamlit wyrenderował dla tego przypadku - pusty
                        // string dla nowego przypadku, żeby żadna resztka
                        // tekstu z poprzedniego przypadku nie została widoczna.
                        nativeSet(el, value);
                    }}
                }});
            }}
            enforceAll();
            if (window.parent.__iscoInstantSaveObserver) {{
                window.parent.__iscoInstantSaveObserver.disconnect();
            }}
            window.parent.__iscoInstantSaveObserver = new MutationObserver(enforceAll);
            window.parent.__iscoInstantSaveObserver.observe(doc.body, {{ childList: true, subtree: true }});
        }})();
        </script>
        """,
        height=0,
    )


def _manual_ctrl_enter_shortcut(idx: int) -> None:
    """Łączy Ctrl/Cmd+Enter z głównym przyciskiem bieżącego kroku Metody A."""
    _ctrl_enter_shortcut(
        handler_key="__manualCtrlEnterHandler",
        fallback_labels=["Zatwierdź kod finalny", "Dalej →"],
        direct_input_aria_label="Pełny kod ISCO-08",
        direct_save_label="Zapisz pełny kod i przejdź dalej",
    )


def _render_selected_vars_caption(row, var_meta: dict, selected_vars: list[str]) -> None:
    """Wyświetla zaznaczone w tabeli 'Dane respondenta' zmienne wraz z etykietą,
    wartością respondenta i (jeśli dostępna) zdekodowaną kategorią - w jednolitym
    formacie używanym we wszystkich modułach (Metoda A, kaskada w modułach B/C)."""
    if not selected_vars:
        st.caption("Brak zaznaczonych zmiennych w tabeli powyżej (kliknij nazwy kolumn, żeby je zaznaczyć).")
        return

    info_lines = []
    for col in selected_vars:
        raw_val = row.get(col)
        value_labels = var_meta.get(col, {}).get("value_labels", {})
        decoded = None
        if pd.notna(raw_val) and value_labels:
            key_candidates = [str(raw_val)]
            try:
                key_candidates.append(str(int(float(raw_val))))
            except (ValueError, TypeError):
                pass
            for k in key_candidates:
                if k in value_labels:
                    decoded = value_labels[k]
                    break
        label = var_meta.get(col, {}).get("label", "")
        line = f"**{col}**"
        if label:
            line += f" _{label}_"
        line += f": wartość respondenta = `{raw_val}`"
        if decoded:
            line += f" → **{decoded}**"
        info_lines.append(line)
    st.caption("Zmienne zaznaczone w tabeli powyżej, użyte przy tej decyzji:  \n" + "  \n".join(info_lines))


def render_manual_step(df, idx: int, row, df_state_key: str = "manual_df", idx_state_key: str = "manual_idx"):
    level = st.session_state.setdefault(f"manual_step_{idx}", 1)
    digits = st.session_state.setdefault(f"manual_digits_{idx}", [])
    prefix = "".join(digits)

    st.info(f"**{LEVEL_LABELS[level]}**" + (f" — dotychczas wybrany prefiks kodu: `{prefix}`" if prefix else ""))

    options_data = _manual_level_options(level, prefix)
    label_to_code = {item["label"]: item["code"] for item in options_data}
    labels = [item["label"] for item in options_data]
    choice = st.radio(
        "Wybierz kod dla tego poziomu",
        options=labels,
        index=None,
        key=f"manual_choice_{idx}_{level}_{prefix}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    target = _get_coding_target(df_state_key)
    n = len(df)
    qualifying_positions = _qualifying_positions(df, target)

    # Zmienne, z których korzystał koder = dowolna kombinacja kolumn
    # zaznaczonych w widocznej tabeli "Dane respondenta" (patrz analogiczna
    # logika w render_cascade_step).
    source_cols = set(visible_df_for_mode(df.iloc[[idx]], target).columns)
    var_meta = load_var_metadata(target)
    table_selection = st.session_state.get(_resp_table_key(idx, df_state_key), {})
    clicked_cols = table_selection.get("selection", {}).get("columns", [])
    selected_vars = [c for c in clicked_cols if c in source_cols]

    _render_selected_vars_caption(row, var_meta, selected_vars)
    _render_previously_used_vars_caption(df, idx, selected_vars)

    st.markdown("**Lub wpisz od razu pełny, 4-cyfrowy kod ISCO-08:**")
    _seed_direct_code_and_uzasadnienie(
        df, idx,
        direct_key=f"manual_direct_code_{idx}",
        uzasadnienie_key=f"manual_uzasadnienie_{idx}",
    )
    direct_code = st.text_input(
        "Pełny kod ISCO-08",
        max_chars=4,
        placeholder="",
        key=f"manual_direct_code_{idx}",
        label_visibility="collapsed",
    ).strip()

    uzasadnienie = st.text_area(
        "Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)",
        key=f"manual_uzasadnienie_{idx}",
        height=70,
    )

    valid_final_codes = set(load_embeddings_level(4)[3])
    valid_level1_codes = set(load_embeddings_level(1)[3])
    valid_level2_codes = set(load_embeddings_level(2)[3])
    valid_level3_codes = set(load_embeddings_level(3)[3])

    if st.button(
        "Zapisz pełny kod i przejdź dalej",
        type="primary",
        use_container_width=True,
        key=f"manual_direct_save_{idx}",
    ):
        if not (len(direct_code) == 4 and direct_code.isdigit()):
            st.warning("Wpisz 4 cyfry kodu ISCO-08.")
        elif not _direct_code_is_valid(direct_code, valid_final_codes, valid_level1_codes, valid_level2_codes, valid_level3_codes):
            st.warning(
                "Podany kod nie występuje na liście kodów ISCO-08 i nie jest poprawnym "
                "prefiksem dopełnionym zerami (np. 5200)."
            )
        else:
            _manual_save_code(
                df, idx, direct_code, target, uzasadnienie, df_state_key,
                idx_state_key, qualifying_positions, selected_vars,
            )
            st.rerun()

    st.caption("Skrót: Ctrl+Enter (na macOS także Cmd+Enter) uruchamia główny przycisk bieżącego kroku.")
    _manual_ctrl_enter_shortcut(idx)
    _instant_text_persistence(idx, [
        ("Pełny kod ISCO-08", direct_code),
        ("Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)", uzasadnienie),
    ])

    col_back, col_next = st.columns(2)
    with col_back:
        if level > 1 and st.button("← Cofnij krok", use_container_width=True, key=f"manual_back_{idx}"):
            digits.pop()
            st.session_state[f"manual_step_{idx}"] = level - 1
            st.session_state[f"hitl_wracal_{idx}"] = True
            st.rerun()

    with col_next:
        next_label = "Zatwierdź kod finalny" if level == 4 else "Dalej →"
        if st.button(next_label, type="primary", use_container_width=True, key=f"manual_next_{idx}_{level}_{prefix}"):
            if choice is None:
                st.warning("Wybierz jedną opcję przed przejściem dalej.")
                return

            chosen_code = label_to_code[choice]
            if chosen_code == "__UNCODABLE__":
                _manual_save_uncodable(
                    df, idx, target, uzasadnienie, df_state_key,
                    idx_state_key, qualifying_positions,
                )
                st.rerun()
            elif chosen_code == "__NO_MATCH__":
                final_code = (prefix + "0" * (5 - level))[:4]
                _manual_save_code(
                    df, idx, final_code, target, uzasadnienie, df_state_key,
                    idx_state_key, qualifying_positions, selected_vars,
                )
                st.rerun()
            elif level < 4:
                digits.append(chosen_code[-1])
                df.at[idx, f"ISCO_poziom{level}"] = chosen_code[-1]
                df.at[idx, f"ISCO_poziom{level}_zmienne"] = ", ".join(selected_vars) if selected_vars else None
                _persist_df(df_state_key, df)
                st.session_state[f"manual_step_{idx}"] = level + 1
                st.rerun()
            else:
                final_code = chosen_code
                _manual_save_code(
                    df, idx, final_code, target, uzasadnienie, df_state_key,
                    idx_state_key, qualifying_positions, selected_vars,
                )
                st.rerun()


def render_classify_manual():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()

    if st.button("← Wróć do strony głównej", key="back_manual"):
        go_to("home")
        st.rerun()

    st.markdown(
        '<h1 style="text-align:center; line-height:1.3;">Metoda A<br>'
        '<span style="font-size:0.6em;">Kodowanie ręczne</span></h1>',
        unsafe_allow_html=True,
    )
    st.write(
        "Wczytaj plik CSV zawierający dane ankietowe European Social Survey (ESS). "
        "Ekspert wybiera kod ISCO-08 ręcznie, poziom po poziomie, z pełnych list "
        "kodów uporządkowanych rosnąco."
    )

    uploaded_file = st.file_uploader("Wybierz plik CSV", type=["csv"], key="uploader_manual")

    if uploaded_file is None:
        # Brak nowo wybranego pliku - to normalne zaraz po odświeżeniu strony
        # (st.file_uploader zawsze wraca jako None po F5). Zanim skasujemy
        # postęp, sprawdzamy, czy nie mamy zapisanego na dysku df z tej samej
        # sesji roboczej (patrz _persist_df / _load_df_cache).
        if "manual_df" not in st.session_state:
            cached_df, cached_source = _load_df_cache("manual_df")
            if cached_df is not None:
                st.session_state["manual_df"] = cached_df
                st.session_state["manual_source"] = cached_source
                resume_target_manual = _get_coding_target("manual_df")
                fallback_idx = _first_unfinished_idx(cached_df, resume_target_manual)
                # Preferujemy DOKŁADNĄ pozycję zapamiętaną w URL (np. przypadek
                # 99, jeśli koder tam właśnie był) nad "pierwszym nieukończonym" -
                # patrz _set_idx / _restore_idx_from_query.
                st.session_state[_frontier_key("manual_idx")] = _restore_frontier_from_query("manual_idx", fallback_idx)
                _set_idx("manual_idx", _restore_idx_from_query("manual_idx", fallback_idx, len(cached_df)))
        if "manual_df" not in st.session_state:
            return

    if uploaded_file is not None and (
        "manual_df" not in st.session_state or st.session_state.get("manual_source") != uploaded_file.name
    ):
        df = read_csv_robust(uploaded_file)
        for col in ("B33", "B34", "B35", "B48", "B49", "B50"):
            if col not in df.columns:
                st.error(f"W pliku nie znaleziono wymaganej kolumny: {col}")
                return

        _init_manual_result_columns(df)
        resume_target_manual = _get_coding_target("manual_df")
        resume_idx = _first_unfinished_idx(df, resume_target_manual)
        _set_idx("manual_idx", resume_idx)
        st.session_state["manual_source"] = uploaded_file.name
        _persist_df("manual_df", df, source_name=uploaded_file.name)
        if 0 < resume_idx < len(df):
            _, resume_rank, resume_total = _qualifying_progress(
                _qualifying_positions(df, resume_target_manual), resume_idx, len(df)
            )
            st.session_state["manual_resume_msg"] = (
                f"Wykryto częściowo wypełniony plik - wznowiono kodowanie od osoby nr {resume_rank} z {resume_total}."
            )
        elif resume_idx >= len(df) and len(df) > 0:
            st.session_state["manual_resume_msg"] = (
                "Wykryto plik, w którym wszystkie osoby w bieżącym trybie są już zakodowane."
            )

    df = st.session_state["manual_df"]
    idx = st.session_state["manual_idx"]
    n = len(df)
    if idx < n:
        _restore_widget_drafts("manual_df", idx)

    resume_msg = st.session_state.pop("manual_resume_msg", None)
    if resume_msg:
        st.info(resume_msg)

    st.write("")
    render_mode_selector("manual_df", "manual_idx", df, idx, n)
    mode_manual = _get_coding_target("manual_df")
    _warn_if_meta_missing(mode_manual)
    st.write("")

    podmiot_label = "Partner" if mode_manual == "Partner" else "Respondent"
    qualifying_positions_manual = _qualifying_positions(df, mode_manual)
    progress_fraction, progress_rank, progress_total = _qualifying_progress(qualifying_positions_manual, idx, n)
    st.progress(progress_fraction, text=f"{podmiot_label} {progress_rank} z {progress_total}")

    _render_case_jumper(
        qualifying_positions_manual, idx, "manual_idx",
        key_suffix=f"manual_{mode_manual}",
        on_jump=_manual_reset_idx,
    )

    mode_suffix = "respondent" if mode_manual == "Respondent" else "partner"

    # Pozwala wrócić do ostatnio zakodowanego przypadku również po ukończeniu
    # całego pliku. Ponowny zapis po prostu nadpisuje poprzednią decyzję.
    previous_manual_idx = _prev_qualifying_idx(qualifying_positions_manual, idx)
    col_prev, col_next = st.columns(2)
    with col_prev:
        if previous_manual_idx != idx:
            if st.button("← Wróć do poprzedniego przypadku", key=f"manual_prev_case_{idx}", use_container_width=True):
                _manual_reset_idx(previous_manual_idx)
                _set_idx("manual_idx", previous_manual_idx)
                st.rerun()
    with col_next:
        if _render_next_case_button(
            qualifying_positions_manual, idx, "manual_idx", key_suffix=f"manual_{mode_manual}", on_jump=_manual_reset_idx
        ):
            st.rerun()

    if idx < n:
        with st.expander(f"Pobierz częściowy wynik (dotychczasowy postęp: {progress_rank - 1} z {progress_total})"):
            unfinished_manual = _unfinished_case_numbers(df, mode_manual, qualifying_positions_manual)
            if unfinished_manual:
                st.caption(
                    f"Nieukończone numery przypadków ({len(unfinished_manual)}): "
                    + ", ".join(str(n) for n in unfinished_manual)
                )
            partial_csv, partial_xlsx = _build_csv_xlsx_bytes(df)
            col_pdl1, col_pdl2 = st.columns(2)
            with col_pdl1:
                st.download_button(
                    "Pobierz częściowy wynik (CSV)",
                    data=partial_csv,
                    file_name=f"wynik_czesciowy_metoda_a_{mode_suffix}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="manual_partial_dl_csv",
                )
            with col_pdl2:
                st.download_button(
                    "Pobierz częściowy wynik (Excel .xlsx)",
                    data=partial_xlsx,
                    file_name=f"wynik_czesciowy_metoda_a_{mode_suffix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="manual_partial_dl_xlsx",
                )

    if idx >= n:
        podmiot_plural = "partnerów" if mode_manual == "Partner" else "respondentów"
        st.success(f"Zakodowano wszystkich {podmiot_plural}.")
        csv_bytes, xlsx_bytes = _build_csv_xlsx_bytes(df)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Pobierz wynik (CSV)",
                data=csv_bytes,
                file_name=f"wynik_metoda_a_{mode_suffix}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Pobierz wynik (Excel .xlsx)",
                data=xlsx_bytes,
                file_name=f"wynik_metoda_a_{mode_suffix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        return

    row = df.iloc[idx]
    st.session_state.setdefault(f"hitl_start_time_{idx}", time.time())
    st.session_state.setdefault(f"hitl_wracal_{idx}", False)
    st.session_state.setdefault(f"manual_step_{idx}", 1)
    st.session_state.setdefault(f"manual_digits_{idx}", [])

    _display_respondent_idno(row)
    st.write("Dane respondenta:")
    st.caption("Kliknij nazwy kolumn (zmiennych), z których korzystasz przy klasyfikacji.")
    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_manual),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, load_var_metadata(mode_manual)),
        on_select="rerun",
        selection_mode=["multi-column"],
        key=_resp_table_key(idx, "manual_df"),
    )

    cols = _target_cols("manual_df")
    with st.container(border=True):
        st.markdown(f"**Zawód:** {row[cols['zawod']]}")
        st.markdown(f"**Obowiązki i zadania:** {row[cols['obowiazki']]}")
        st.markdown(f"**Wykształcenie:** {row[cols['wyksztalcenie']]}")

    render_manual_step(df, idx, row)
    _persist_widget_drafts("manual_df", idx)


# ============================================================
# STRONA: KLASYFIKACJA Z UDZIAŁEM EKSPERTA (1 CYFRA PRZYPORZĄDKOWANA)
# ============================================================
# Kolumna z danych wejściowych, w której przechowywany jest kod ISCO-08
# wcześniej przyporządkowany respondentowi (np. przez system automatyczny),
# ale TYLKO dla 1. cyfry - pozostałe cyfry (2, 3, 4) nie są przyporządkowane
# i są dokodowywane tak samo jak w pełnej kaskadzie (AI proponuje kandydatów).
PA_DIGIT_COLUMNS = {1: "ISCO_1Digit_Respondent"}


def _normalize_isco_digit_code(raw_value, level: int) -> Optional[str]:
    """Zamienia wartość z kolumny ISCO_{level}Digit_Respondent (np. wczytaną
    jako 7, 7.0 albo '07') na czysty, L-cyfrowy string kodu. Zwraca None,
    jeśli wartość jest pusta lub niepoprawna (np. za krótka/za długa)."""
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.zfill(level)
    if len(text) != level or not text.isdigit():
        return None
    return text


def render_pa_digit1_step(df, idx: int, row, df_state_key: str, idx_state_key: str):
    """Renderuje krok potwierdzenia przyporządkowanej 1. cyfry kodu ISCO-08.
    Po zatwierdzeniu respondent przechodzi w tryb kaskady od razu na poziomie 2
    (prefiks = zatwierdzona 1. cyfra) - cyfry 2-4 są dokodowywane przez AI
    dokładnie tak samo jak w pełnym module kaskadowym (render_cascade_step).
    Po odrzuceniu respondent wraca do pełnego kodowania kaskadowego od 1. cyfry."""
    col_name = PA_DIGIT_COLUMNS[1]
    proposed_code = _normalize_isco_digit_code(row.get(col_name), 1)

    st.info("**Weryfikacja 1. cyfry (grupa główna)**")

    if proposed_code is None:
        st.warning(
            f"Brak poprawnie przyporządkowanej cyfry w kolumnie `{col_name}` dla tego "
            "respondenta - przechodzę do pełnego kodowania kaskadowego od 1. cyfry."
        )
        _start_cascade(idx)
        st.rerun()
        return

    _, _, _, _, metadata = load_embeddings_level(1)
    proposed_title = metadata.get(proposed_code, {}).get("title", "")
    proposed_title_en = metadata.get(proposed_code, {}).get("title_en", "")
    title_txt = f" — {_lower_first(proposed_title)}" if proposed_title else ""
    if proposed_title_en:
        title_txt += f" (ang. {proposed_title_en})"
    st.markdown(f"**Przyporządkowana cyfra:** `{proposed_code}`{title_txt}")

    decision = st.radio(
        "Czy zatwierdzasz tę cyfrę?",
        options=["Tak, zatwierdzam", "Nie, nie zgadzam się"],
        key=f"pa1_decision_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    komentarz = ""
    if decision == "Nie, nie zgadzam się":
        komentarz = st.text_area(
            "Proszę opisać powód odrzucenia przyporządkowanej cyfry (opcjonalnie)",
            key=f"pa1_komentarz_{idx}",
            height=70,
        )

    is_confirm = decision == "Tak, zatwierdzam"
    button_label = "Zatwierdź i pokaż dopasowane kody" if is_confirm else "Odrzuć i koduj od nowa (kaskadowo)"

    target = _get_coding_target(df_state_key)
    qualifying_positions = _qualifying_positions(df, target)

    col_next = st.container()
    with col_next:
        if st.button(button_label, type="primary", use_container_width=True, key=f"pa1_next_{idx}"):
            df.at[idx, "Cyfra1_zatwierdzona_expert"] = "Tak" if is_confirm else "Nie"
            if not is_confirm and komentarz.strip():
                df.at[idx, "Powod_odrzucenia_cyfry"] = komentarz.strip()
            _persist_df(df_state_key, df)

            if is_confirm:
                df.at[idx, "ISCO_poziom1"] = proposed_code
                st.session_state[f"pa1_confirmed_{idx}"] = proposed_code
            else:
                st.session_state[f"hitl_wracal_{idx}"] = True
                _start_cascade(idx)
            st.rerun()


def render_pa_top10_step(df, idx: int, row, prefix: str, df_state_key: str, idx_state_key: str):
    """Po zatwierdzeniu 1. cyfry pokazuje TOP-10 najlepiej dopasowanych pełnych
    (4-cyfrowych) kodów ISCO-08, zawężonych do tych zaczynających się od `prefix`
    (potwierdzona 1. cyfra) - analogicznie do widoku top-10 w module 3. Ekspert
    może wybrać jeden z nich albo przejść do kodowania kaskadowego cyfr 2-4."""
    st.info(f"**1. cyfra zatwierdzona: `{prefix}`** — poniżej najlepiej dopasowane pełne kody ISCO-08")

    target = _get_coding_target(df_state_key)
    cols = _target_cols(df_state_key)
    n = len(df)
    qualifying_positions = _qualifying_positions(df, target)

    model = load_model()
    title_emb, tasks_emb, synteza_emb, codes_ordered, metadata = load_embeddings()

    cache_key = f"pa_top10_candidates_{idx}_{target}"
    if cache_key not in st.session_state:
        ranking = classify(
            zawod_czlowieka=str(row[cols["zawod"]]) if pd.notna(row[cols["zawod"]]) else "",
            umiejetnosci_obowiazki=str(row[cols["obowiazki"]]) if pd.notna(row[cols["obowiazki"]]) else "",
            wyksztalcenie=str(row[cols["wyksztalcenie"]]) if pd.notna(row[cols["wyksztalcenie"]]) else "",
            model=model,
            title_emb=title_emb,
            tasks_emb=tasks_emb,
            synteza_emb=synteza_emb,
            codes_ordered=codes_ordered,
            metadata=metadata,
            top_k=10,
            prefix=prefix,
        )
        st.session_state[cache_key] = ranking

    ranking = st.session_state[cache_key]

    NO_CODE_OPTION = "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)"
    fill_code = f"{prefix}000"
    NO_DETERMINATION_OPTION = (
        f"Brak możliwości ustalenia dokładnej cyfry (dopełnij pozostałe cyfry zerami, kod: {fill_code})"
    )
    options = [_format_candidate_label(r.isco_code, r.title_pl, getattr(r, "title_en", ""), r.score) for r in ranking.itertuples()]
    options.append(NO_DETERMINATION_OPTION)
    options.append(NO_CODE_OPTION)

    _seed_top10_widgets(
        df, idx, options,
        radio_key=f"pa_top10_choice_{idx}",
        direct_key=f"pa_top10_direct_code_{idx}",
        uzasadnienie_key=f"pa_top10_uzasadnienie_{idx}",
    )

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        index=None,
        key=f"pa_top10_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None
    is_uncodable = choice == NO_CODE_OPTION
    is_no_determination = choice == NO_DETERMINATION_OPTION

    if choice is None:
        chosen_code = None
    elif is_no_determination:
        chosen_code = fill_code
    elif is_uncodable:
        chosen_code = None
    else:
        choice_idx = options.index(choice)
        chosen_code = ranking.iloc[choice_idx]["isco_code"]

    if st.button(
        "**← Cofnij do wyboru 1. cyfry**",
        key=f"pa_top10_back_to_digit1_{idx}",
        use_container_width=True,
    ):
        # Cofa zatwierdzenie 1. cyfry - respondent wraca do render_pa_digit1_step
        # (patrz warunek `if pa1_confirmed:` w render_classify_hitl_1digit),
        # gdzie można ponownie zatwierdzić albo odrzucić przyporządkowaną cyfrę.
        df.at[idx, "Cyfra1_zatwierdzona_expert"] = None
        _persist_df(df_state_key, df)
        st.session_state.pop(f"pa1_confirmed_{idx}", None)
        st.session_state.pop(cache_key, None)
        st.rerun()

    # Ocena pomocności listy 10 dopasowanych kodów - wymagana zawsze, niezależnie
    # od tego, czy koder wybierze jeden z nich, czy przejdzie do kodowania
    # kaskadowego (patrz przycisk "Kontynuuj kodowanie kaskadowo" niżej).
    ai_helpfulness = render_helpfulness_scale(
        "Jak pomocne były dopasowane kody? (1-5 punktów)",
        key=f"pa_top10_ai_helpfulness_{idx}",
    )

    st.markdown("**Lub wpisz od razu pełny, 4-cyfrowy kod ISCO-08:**")
    direct_code = st.text_input(
        "Pełny kod ISCO-08",
        max_chars=4,
        placeholder="",
        key=f"pa_top10_direct_code_{idx}",
        label_visibility="collapsed",
    ).strip()

    uzasadnienie_top10 = st.text_area(
        "Uzasadnienie / komentarz do wyboru (opcjonalnie)",
        key=f"pa_top10_uzasadnienie_{idx}",
        height=70,
    )

    valid_final_codes = set(load_embeddings_level(4)[3])
    valid_level1_codes = set(load_embeddings_level(1)[3])
    valid_level2_codes = set(load_embeddings_level(2)[3])
    valid_level3_codes = set(load_embeddings_level(3)[3])

    col_btn1 = st.container()
    with col_btn1:
        if st.button("Zapisz i przejdź dalej", type="primary", use_container_width=True, key=f"pa_top10_save_{idx}"):
            if direct_code:
                # Ręcznie wpisany kod ma pierwszeństwo przed zaznaczeniem na liście
                # radio - koder mógł zaznaczyć jakąś opcję wcześniej, a potem
                # zmienić zdanie i wpisać kod bezpośrednio.
                if not (len(direct_code) == 4 and direct_code.isdigit()):
                    st.warning("Wpisz 4 cyfry kodu ISCO-08.")
                elif not _direct_code_is_valid(direct_code, valid_final_codes, valid_level1_codes, valid_level2_codes, valid_level3_codes):
                    st.warning(
                        "Podany kod nie występuje na liście kodów ISCO-08 i nie jest poprawnym "
                        "prefiksem dopełnionym zerami (np. 5200)."
                    )
                else:
                    df.at[idx, "ISCO_wybrany"] = direct_code
                    df.at[idx, "ISCO_poziom1"] = direct_code[0]
                    df.at[idx, "ISCO_poziom2"] = direct_code[1]
                    df.at[idx, "ISCO_poziom3"] = direct_code[2]
                    df.at[idx, "ISCO_poziom4"] = direct_code[3]
                    df.at[idx, "ISCO_PRED"] = direct_code
                    if uzasadnienie_top10.strip():
                        df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                    df.at[idx, "Brak_mozliwosci_zakodowania"] = None
                    df.at[idx, "Kodowany_podmiot"] = target
                    _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key=df_state_key)
                    _persist_df(df_state_key, df)
                    st.session_state.pop(f"pa1_confirmed_{idx}", None)
                    st.session_state.pop(cache_key, None)
                    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                    st.rerun()
            elif is_uncodable:
                df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key=df_state_key)
                _persist_df(df_state_key, df)
                st.session_state.pop(f"pa1_confirmed_{idx}", None)
                st.session_state.pop(cache_key, None)
                _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                st.rerun()
            elif choice is None:
                st.warning("Wybierz jedną opcję z listy albo wpisz kod ręcznie przed zapisaniem.")
            else:
                df.at[idx, "ISCO_wybrany"] = chosen_code
                df.at[idx, "Decyzja_kodera_zawod"] = decyzja_kodera_zawod
                df.at[idx, "Decyzja_kodera_notatka"] = decyzja_kodera_notatka
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                if chosen_code:
                    df.at[idx, "ISCO_poziom1"] = chosen_code[0]
                    df.at[idx, "ISCO_poziom2"] = chosen_code[1]
                    df.at[idx, "ISCO_poziom3"] = chosen_code[2]
                    df.at[idx, "ISCO_poziom4"] = chosen_code[3]
                    df.at[idx, "ISCO_PRED"] = chosen_code
                rank, score = _get_rank_and_score(ranking, chosen_code)
                df.at[idx, "Ranking_pozycja_wybranego_kodu"] = rank
                df.at[idx, "Score_wybranego_kodu"] = score
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key=df_state_key)
                _persist_df(df_state_key, df)
                st.session_state.pop(f"pa1_confirmed_{idx}", None)
                st.session_state.pop(cache_key, None)
                _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                st.rerun()

    st.caption("Skrót: Ctrl+Enter (na macOS także Cmd+Enter) zapisuje wybór i przechodzi dalej.")
    _ctrl_enter_shortcut(
        handler_key="__paTop10CtrlEnterHandler",
        fallback_labels=["Zapisz i przejdź dalej"],
    )
    _instant_text_persistence(idx, [
        ("Pełny kod ISCO-08", direct_code),
        ("Uzasadnienie / komentarz do wyboru (opcjonalnie)", uzasadnienie_top10),
    ])

    st.write("")
    if st.button(
        "Kontynuuj kodowanie kaskadowo (cyfry 2-4)",
        key=f"pa_cascade_start_{idx}",
        use_container_width=True,
    ):
        # Zapisujemy ocenę pomocności listy 10 kodów, zanim koder przejdzie
        # do kodowania kaskadowego - inaczej ta ocena nigdy by się nie zapisała.
        df.at[idx, "Ocena_AI_top10_1_5"] = ai_helpfulness
        _persist_df(df_state_key, df)
        st.session_state.pop(f"pa1_confirmed_{idx}", None)
        st.session_state.pop(cache_key, None)
        # Jeśli ten przypadek ma już zapisane kolejne cyfry (poziom 2-3) z
        # wcześniejszej decyzji - np. koder wrócił, żeby coś poprawić - wizard
        # wznawia się OD RAZU za ostatnią zapisaną cyfrą, zamiast zawsze od
        # cyfry 2. Nigdy nie sięga do poziomu 4 (poza 1-4 nie ma sensu tutaj).
        resume_digits = [prefix]
        for level_col in ("ISCO_poziom2", "ISCO_poziom3"):
            val = df.at[idx, level_col] if level_col in df.columns else None
            if isinstance(val, str) and val.strip().isdigit():
                resume_digits.append(val.strip())
            else:
                break
        st.session_state[f"cascade_step_{idx}"] = len(resume_digits) + 1
        st.session_state[f"cascade_digits_{idx}"] = resume_digits
        st.rerun()


def render_classify_hitl_1digit():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()

    if st.button("← Wróć do strony głównej", key="back_hitl_1digit"):
        go_to("home")
        st.rerun()

    st.markdown(
        '<h1 style="text-align:center; line-height:1.3;">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta'
        '<br><span style="font-size:0.6em;">(1&nbsp;cyfra&nbsp;przyporządkowana)</span></h1>',
        unsafe_allow_html=True,
    )
    st.write(
        "Wczytaj plik CSV z danymi ankietowymi European Social Survey (ESS), zawierający "
        "wstępnie przypisaną pierwszą cyfrę kodu ISCO-08. Dla każdego respondenta lub jego "
        "partnera zatwierdź lub odrzuć zaproponowaną pierwszą cyfrę. Po jej zatwierdzeniu "
        "system wyświetli 10 najbardziej prawdopodobnych pełnych kodów ISCO-08 ograniczonych "
        "do wybranej grupy głównej. Jeżeli żadna z propozycji nie okaże się właściwa, możliwe "
        "jest przejście do kodowania kaskadowego rozpoczynającego się od drugiego poziomu "
        "klasyfikacji. W przypadku odrzucenia pierwszej cyfry proces rozpoczyna się od "
        "początku, czyli od wyboru pierwszego poziomu klasyfikacji ISCO-08."
    )

    uploaded_file = st.file_uploader("Wybierz plik CSV", type=["csv"], key="uploader_hitl_1digit")

    if uploaded_file is None:
        # Po odświeżeniu strony uploaded_file zawsze wraca jako None - próbujemy
        # najpierw odtworzyć df z cache na dysku (patrz _load_df_cache), zanim
        # skasujemy postęp kodowania.
        if "hitl1d_df" not in st.session_state:
            cached_df, cached_source = _load_df_cache("hitl1d_df")
            if cached_df is not None:
                st.session_state["hitl1d_df"] = cached_df
                st.session_state["hitl1d_source"] = cached_source
                resume_target_1d = _get_coding_target("hitl1d_df")
                fallback_idx = _first_unfinished_idx(cached_df, resume_target_1d)
                st.session_state[_frontier_key("hitl1d_idx")] = _restore_frontier_from_query("hitl1d_idx", fallback_idx)
                _set_idx("hitl1d_idx", _restore_idx_from_query("hitl1d_idx", fallback_idx, len(cached_df)))
        if "hitl1d_df" not in st.session_state:
            return

    if uploaded_file is not None and (
        "hitl1d_df" not in st.session_state or st.session_state.get("hitl1d_source") != uploaded_file.name
    ):
        df = read_csv_robust(uploaded_file)

        required_cols = ["B33", "B34", "B35", "B48", "B49", "B50"] + list(PA_DIGIT_COLUMNS.values())
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error("W pliku brakuje wymaganych kolumn: " + ", ".join(missing))
            return

        # Kolejność kolumn dodawanych do wyniku jest ujednolicona z modułem 1
        # (Klasyfikacja zawodów z udziałem eksperta) - te same grupy w tej samej
        # kolejności. Dwie kolumny właściwe tylko dla tego modułu
        # (Cyfra1_zatwierdzona_expert, Powod_odrzucenia_cyfry - dotyczą decyzji
        # o wstępnie przypisanej 1. cyfrze) są wstawione zaraz po ISCO_PRED,
        # bo logicznie poprzedzają dalsze etapy kodowania.
        if "ISCO_wybrany" not in df.columns:
            df["ISCO_wybrany"] = None
            df["Decyzja_kodera_zawod"] = None
            df["Decyzja_kodera_notatka"] = None
        if "Kodowany_podmiot" not in df.columns:
            df["Kodowany_podmiot"] = None
        if "Brak_mozliwosci_zakodowania" not in df.columns:
            df["Brak_mozliwosci_zakodowania"] = None
        for col in ("ISCO_poziom1", "ISCO_poziom2", "ISCO_poziom3", "ISCO_poziom4", "ISCO_PRED"):
            if col not in df.columns:
                df[col] = None
        for col in ("Cyfra1_zatwierdzona_expert", "Powod_odrzucenia_cyfry"):
            if col not in df.columns:
                df[col] = None
        for col in (
            "ISCO_poziom1_zmienne",
            "ISCO_poziom2_zmienne",
            "ISCO_poziom3_zmienne",
            "ISCO_poziom4_zmienne",
            "ISCO_poziom1_ranking_pozycja",
            "ISCO_poziom1_score",
            "ISCO_poziom2_ranking_pozycja",
            "ISCO_poziom2_score",
            "ISCO_poziom3_ranking_pozycja",
            "ISCO_poziom3_score",
            "ISCO_poziom4_ranking_pozycja",
            "ISCO_poziom4_score",
            "Ranking_pozycja_wybranego_kodu",
            "Score_wybranego_kodu",
            "Uzasadnienie_finalne",
            "Czas_kodowania_sekundy",
            "Czas_do_pierwszej_interakcji_sekundy",
            "Czy_uzytkownik_wracal",
            "Ocena_AI_top10_1_5",
            "Ocena_AI_kaskadowo_1_5",
        ):
            if col not in df.columns:
                df[col] = None

        # Wymuszenie właściwego dtype (patrz _ensure_text_column_dtype) - kluczowe
        # przy WZNOWIENIU kodowania z częściowo wypełnionego pliku CSV, w którym
        # pandas mógł błędnie nadać kolumnom z kodami ISCO-08 typ float64.
        for col in (
            "ISCO_wybrany",
            "Decyzja_kodera_zawod",
            "Decyzja_kodera_notatka",
            "Kodowany_podmiot",
            "Brak_mozliwosci_zakodowania",
            "ISCO_poziom1",
            "ISCO_poziom2",
            "ISCO_poziom3",
            "ISCO_poziom4",
            "ISCO_PRED",
            "Cyfra1_zatwierdzona_expert",
            "Powod_odrzucenia_cyfry",
            "ISCO_poziom1_zmienne",
            "ISCO_poziom2_zmienne",
            "ISCO_poziom3_zmienne",
            "ISCO_poziom4_zmienne",
            "Uzasadnienie_finalne",
        ):
            _ensure_text_column_dtype(df, col)
        _ensure_object_dtype(df, "Czy_uzytkownik_wracal")


        resume_target_1d = _get_coding_target("hitl1d_df")
        resume_idx = _first_unfinished_idx(df, resume_target_1d)
        _set_idx("hitl1d_idx", resume_idx)
        st.session_state["hitl1d_source"] = uploaded_file.name
        _persist_df("hitl1d_df", df, source_name=uploaded_file.name)
        if 0 < resume_idx < len(df):
            _, resume_rank, resume_total = _qualifying_progress(
                _qualifying_positions(df, resume_target_1d), resume_idx, len(df)
            )
            st.session_state["hitl1d_resume_msg"] = (
                f"Wykryto częściowo wypełniony plik - wznowiono kodowanie od osoby nr {resume_rank} z {resume_total}."
            )
        elif resume_idx >= len(df) and len(df) > 0:
            st.session_state["hitl1d_resume_msg"] = (
                "Wykryto plik, w którym wszystkie osoby w bieżącym trybie są już zakodowane."
            )

    df = st.session_state["hitl1d_df"]
    idx = st.session_state["hitl1d_idx"]
    n = len(df)
    if idx < n:
        _restore_widget_drafts("hitl1d_df", idx)

    resume_msg = st.session_state.pop("hitl1d_resume_msg", None)
    if resume_msg:
        st.info(resume_msg)
    st.write("")
    render_mode_selector("hitl1d_df", "hitl1d_idx", df, idx, n)
    mode_1digit = _get_coding_target("hitl1d_df")
    _warn_if_meta_missing(mode_1digit)
    st.write("")
    podmiot_label = "Partner" if mode_1digit == "Partner" else "Respondent"
    qualifying_positions_1d = _qualifying_positions(df, mode_1digit)
    progress_fraction, progress_rank, progress_total = _qualifying_progress(qualifying_positions_1d, idx, n)
    st.progress(progress_fraction, text=f"{podmiot_label} {progress_rank} z {progress_total}")

    _render_case_jumper(qualifying_positions_1d, idx, "hitl1d_idx", key_suffix=f"hitl1d_{mode_1digit}")

    previous_1d_idx = _prev_qualifying_idx(qualifying_positions_1d, idx)
    prev_label_1d_top = "← Poprzedni partner" if mode_1digit == "Partner" else "← Poprzedni respondent"
    col_prev_1d_top, col_next_1d_top = st.columns(2)
    with col_prev_1d_top:
        if previous_1d_idx != idx:
            if st.button(prev_label_1d_top, key=f"hitl1d_prev_case_top_{idx}", use_container_width=True):
                st.session_state[f"hitl_wracal_{previous_1d_idx}"] = True
                st.session_state.pop(f"pa1_confirmed_{idx}", None)
                _set_idx("hitl1d_idx", previous_1d_idx)
                st.rerun()
    with col_next_1d_top:
        if _render_next_case_button(qualifying_positions_1d, idx, "hitl1d_idx", key_suffix=f"hitl1d_top_{mode_1digit}"):
            st.rerun()

    mode_suffix = "respondent" if mode_1digit == "Respondent" else "partner"

    # Pobranie CZĘŚCIOWEGO wyniku - dostępne cały czas w trakcie kodowania
    # (nie trzeba czekać na ukończenie całego pliku). Format pliku jest
    # identyczny jak wynik końcowy, więc częściowy plik można bez przeszkód
    # wgrać z powrotem później - aplikacja sama wznowi kodowanie od pierwszej
    # nieukończonej osoby (patrz _first_unfinished_idx).
    if idx < n:
        with st.expander(f"Pobierz częściowy wynik (dotychczasowy postęp: {progress_rank - 1} z {progress_total})"):
            unfinished_1d = _unfinished_case_numbers(df, mode_1digit, qualifying_positions_1d)
            if unfinished_1d:
                st.caption(
                    f"Nieukończone numery przypadków ({len(unfinished_1d)}): "
                    + ", ".join(str(n) for n in unfinished_1d)
                )
            partial_csv, partial_xlsx = _build_csv_xlsx_bytes(df)
            col_pdl1, col_pdl2 = st.columns(2)
            with col_pdl1:
                st.download_button(
                    "Pobierz częściowy wynik (CSV)",
                    data=partial_csv,
                    file_name=f"wynik_czesciowy_weryfikacji_1digit_{mode_suffix}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="hitl1d_partial_dl_csv",
                )
            with col_pdl2:
                st.download_button(
                    "Pobierz częściowy wynik (Excel .xlsx)",
                    data=partial_xlsx,
                    file_name=f"wynik_czesciowy_weryfikacji_1digit_{mode_suffix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="hitl1d_partial_dl_xlsx",
                )

    if idx >= n:
        podmiot_plural = "partnerów" if mode_1digit == "Partner" else "respondentów"
        st.success(f"Zweryfikowano wszystkich {podmiot_plural}.")

        csv_bytes, xlsx_bytes = _build_csv_xlsx_bytes(df)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Pobierz wynik (CSV)",
                data=csv_bytes,
                file_name=f"wynik_weryfikacji_1digit_{mode_suffix}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Pobierz wynik (Excel .xlsx)",
                data=xlsx_bytes,
                file_name=f"wynik_weryfikacji_1digit_{mode_suffix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        return

    row = df.iloc[idx]

    st.session_state.setdefault(f"hitl_start_time_{idx}", time.time())
    st.session_state.setdefault(f"hitl_wracal_{idx}", False)

    _display_respondent_idno(row)
    st.write("Dane respondenta:")
    st.caption("Kliknij nazwy kolumn (zmiennych), z których korzystasz przy klasyfikacji.")

    mode_1d_row = _get_coding_target("hitl1d_df")
    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_1d_row),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, load_var_metadata(mode_1d_row)),
        on_select="rerun",
        selection_mode=["multi-column"],
        key=_resp_table_key(idx, "hitl1d_df"),
    )

    cols = _target_cols("hitl1d_df")

    with st.container(border=True):
        st.markdown(f"**Zawód:** {row[cols['zawod']]}")
        st.markdown(f"**Obowiązki i zadania:** {row[cols['obowiazki']]}")
        st.markdown(f"**Wykształcenie:** {row[cols['wyksztalcenie']]}")

    cascade_active = f"cascade_step_{idx}" in st.session_state
    pa1_confirmed = st.session_state.get(f"pa1_confirmed_{idx}")

    if cascade_active:
        render_cascade_step(df, idx, row, df_state_key="hitl1d_df", idx_state_key="hitl1d_idx")
        _persist_widget_drafts("hitl1d_df", idx)
        return

    if pa1_confirmed:
        render_pa_top10_step(df, idx, row, prefix=pa1_confirmed, df_state_key="hitl1d_df", idx_state_key="hitl1d_idx")
        _persist_widget_drafts("hitl1d_df", idx)
        return

    render_pa_digit1_step(df, idx, row, df_state_key="hitl1d_df", idx_state_key="hitl1d_idx")
    _persist_widget_drafts("hitl1d_df", idx)


# ============================================================
# TRYB KASKADOWY (kodowanie ISCO-08 od zera, cyfra po cyfrze)
# ============================================================
LEVEL_LABELS = {
    1: "Krok 1 z 4 - grupa główna (1. cyfra)",
    2: "Krok 2 z 4 - grupa drugorzędna (2. cyfra)",
    3: "Krok 3 z 4 - grupa średnia (3. cyfra)",
    4: "Krok 4 z 4 - grupa elementarna (4. cyfra, kod finalny)",
}


def _start_cascade(idx: int, df=None) -> None:
    """Rozpoczyna (albo wznawia) kaskadowe kodowanie dla przypadku `idx`.

    Jeśli przekazano `df` i przypadek ma już zapisane wcześniejsze cyfry
    (kolumny ISCO_poziom1-3) - np. koder wrócił do wcześniej zakodowanego
    przypadku i chce doprecyzować/poprawić decyzję - wizard startuje od razu
    na poziomie ZA OSTATNIĄ zapisaną cyfrą, z prefiksem już ustawionym,
    zamiast zawsze zaczynać od poziomu 1. Celowo bierze pod uwagę tylko
    poziomy 1-3 (nigdy 4) - level w kaskadzie musi mieścić się w 1-4
    (LEVEL_LABELS / load_embeddings_level), więc nie ustawiamy tu poziomu 5."""
    digits: list[str] = []
    if df is not None:
        for level_col in ("ISCO_poziom1", "ISCO_poziom2", "ISCO_poziom3"):
            if level_col not in df.columns:
                break
            val = df.at[idx, level_col]
            if isinstance(val, str) and val.strip().isdigit():
                digits.append(val.strip())
            else:
                break
    st.session_state[f"cascade_step_{idx}"] = len(digits) + 1
    st.session_state[f"cascade_digits_{idx}"] = digits


def _cancel_cascade(idx: int):
    st.session_state.pop(f"cascade_step_{idx}", None)
    st.session_state.pop(f"cascade_digits_{idx}", None)


def _cascade_save_direct_code(
    df,
    idx: int,
    final_code: str,
    target: str,
    uzasadnienie: str,
    df_state_key: str,
    idx_state_key: str,
    qualifying_positions: list[int],
) -> None:
    """Zapisuje pełny kod wpisany ręcznie w trakcie kodowania kaskadowego
    (pomijając pozostałe, jeszcze nieprzebyte kroki kaskady) i przechodzi
    do następnego kwalifikującego się przypadku. Analogiczne do
    _manual_save_code używanego w Metodzie A."""
    for level, digit in enumerate(final_code, start=1):
        df.at[idx, f"ISCO_poziom{level}"] = digit
    df.at[idx, "ISCO_PRED"] = final_code
    df.at[idx, "ISCO_wybrany"] = final_code
    df.at[idx, "Kodowany_podmiot"] = target
    df.at[idx, "Brak_mozliwosci_zakodowania"] = None
    if uzasadnienie.strip():
        df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie.strip()
    _save_respondent_meta(df, idx, df_state_key=df_state_key)
    _persist_df(df_state_key, df)
    _cancel_cascade(idx)
    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, len(df)))


def _mark_first_interaction(idx: int):
    """Callback (on_change) na widgecie radio z kandydatami - zapisuje moment
    PIERWSZEGO dotknięcia listy kandydatów przez kodera (proxy na namysł).
    Wywoływane tylko raz - kolejne zmiany selekcji już nic nie nadpisują."""
    key = f"hitl_first_interaction_time_{idx}"
    if key not in st.session_state:
        st.session_state[key] = time.time()


def _build_csv_xlsx_bytes(df: pd.DataFrame) -> tuple[bytes, bytes]:
    """Konwertuje dany DataFrame na bajty CSV (utf-8-sig, żeby polskie znaki
    poprawnie otwierały się w Excelu) i XLSX (z automatycznym dopasowaniem
    szerokości kolumn). Używane zarówno przy pobieraniu WYNIKU KOŃCOWEGO
    (po ukończeniu kodowania), jak i przy pobraniu CZĘŚCIOWEGO postępu w
    dowolnym momencie (patrz przycisk "Pobierz częściowy wynik" widoczny
    cały czas podczas kodowania) - to ten sam format pliku w obu przypadkach,
    więc częściowy plik można bez przeszkód wgrać z powrotem później i
    aplikacja sama wznowi kodowanie od pierwszej nieukończonej osoby
    (patrz _first_unfinished_idx)."""
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="wyniki")
        worksheet = writer.sheets["wyniki"]
        for i, col in enumerate(df.columns, start=1):
            max_len = max(
                df[col].apply(lambda v: len(str(v)) if pd.notna(v) else 0).max() if len(df) else 0,
                len(str(col)),
            )
            worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len + 2, 60)
    xlsx_bytes = xlsx_buffer.getvalue()

    return csv_bytes, xlsx_bytes


def _get_rank_and_score(candidates: pd.DataFrame, chosen_code) -> Tuple[Optional[int], Optional[float]]:
    """Zwraca (pozycja_w_rankingu_1_indexed, score) wybranego kodu względem
    listy kandydatów pokazanej koderowi. None, gdy kod nie pochodzi z listy
    (np. wybrano "brak poprawnego kodu" / "brak możliwości ustalenia")."""
    if chosen_code is None:
        return None, None
    matches = candidates.index[candidates["isco_code"] == chosen_code].tolist()
    if not matches:
        return None, None
    pos = matches[0]
    rank = int(pos) + 1
    score = float(candidates.loc[pos, "score"])
    return rank, score


def _get_row_idno(row: pd.Series) -> Optional[str]:
    """Zwraca znormalizowaną wartość IDNO (jako string, bez '.0') niezależnie
    od wielkości liter nazwy kolumny i typu danych - do użytku przy zapisie
    zdarzeń kodowania (patrz _record_coding_event)."""
    idno_col = next((col for col in row.index if str(col).strip().lower() == "idno"), None)
    if idno_col is None or pd.isna(row[idno_col]):
        return None
    value = row[idno_col]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _save_respondent_meta(
    df,
    idx: int,
    ai_helpfulness: Optional[int] = None,
    ai_column: Optional[str] = None,
    df_state_key: Optional[str] = None,
):
    """Zapisuje czas kodowania (sekundy), czas do pierwszej interakcji z listą
    kandydatów, czy koder wracał (cofał się) i ocenę przydatności AI (1-5) -
    w kolumnie zależnej od trybu, którym koder faktycznie kodował tego
    respondenta (ai_column: "Ocena_AI_top10_1_5" albo "Ocena_AI_kaskadowo_1_5").

    Jeśli podano `df_state_key`, dodatkowo zapisuje na bieżąco fakt zakodowania
    tego przypadku do serwerowej ewidencji (patrz _record_coding_event) - dotyczy
    to każdego modułu (A/B/C), bo ta funkcja jest wywoływana przy KAŻDYM
    finalnym zapisie decyzji, niezależnie od ścieżki (ręczne, kaskada, top10)."""
    start = st.session_state.get(f"hitl_start_time_{idx}", time.time())
    now = time.time()
    elapsed = round(now - start, 1)
    df.at[idx, "Czas_kodowania_sekundy"] = elapsed

    first_interaction = st.session_state.get(f"hitl_first_interaction_time_{idx}", now)
    df.at[idx, "Czas_do_pierwszej_interakcji_sekundy"] = round(first_interaction - start, 1)

    df.at[idx, "Czy_uzytkownik_wracal"] = bool(st.session_state.get(f"hitl_wracal_{idx}", False))
    if ai_helpfulness is not None and ai_column is not None:
        df.at[idx, ai_column] = ai_helpfulness
    st.session_state.pop(f"hitl_start_time_{idx}", None)
    st.session_state.pop(f"hitl_wracal_{idx}", None)
    st.session_state.pop(f"hitl_first_interaction_time_{idx}", None)

    if df_state_key is not None:
        wariant = DF_STATE_KEY_TO_WARIANT.get(df_state_key)
        idno = _get_row_idno(df.loc[idx])
        osoba = df.at[idx, "Kodowany_podmiot"] if "Kodowany_podmiot" in df.columns else None
        if wariant is not None and idno is not None and osoba:
            koder = st.session_state.get("username", "")
            _record_coding_event(
                idno=idno,
                osoba=str(osoba),
                wariant=wariant,
                koder=koder,
                isco_kod=df.at[idx, "ISCO_wybrany"] if "ISCO_wybrany" in df.columns else None,
            )
            _record_coding_details(
                idno=idno,
                osoba=str(osoba),
                wariant=wariant,
                koder=koder,
                row=df.loc[idx],
            )


def render_cascade_step(df, idx: int, row, df_state_key: str = "hitl_df", idx_state_key: str = "hitl_idx"):
    """Renderuje pojedynczy krok kaskadowego kodowania dla respondenta `idx`.
    Zwraca True, jeśli kodowanie zostało w tym kroku zakończone i zapisane
    (czyli wywołujący ma przejść do kolejnego respondenta).

    `df_state_key` / `idx_state_key` pozwalają korzystać z tej samej funkcji
    w różnych modułach (np. moduł 3 - kodowanie od zera, moduł 2 - fallback po
    odrzuceniu przyporządkowanej cyfry), każdy trzymający dane pod innym
    kluczem w st.session_state."""
    level = st.session_state[f"cascade_step_{idx}"]
    digits = st.session_state[f"cascade_digits_{idx}"]
    prefix = "".join(digits) if digits else None

    st.info(f"**{LEVEL_LABELS[level]}**" + (f" — dotychczas wybrany prefiks kodu: `{prefix}`" if prefix else ""))

    target = _get_coding_target(df_state_key)
    cols = _target_cols(df_state_key)
    n = len(df)
    qualifying_positions = _qualifying_positions(df, target)

    model = load_model()
    title_emb, tasks_emb, synteza_emb, codes_ordered, metadata = load_embeddings_level(level)

    cache_key = f"cascade_candidates_{idx}_{level}_{prefix}_{target}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = classify_level(
            zawod_czlowieka=str(row[cols["zawod"]]) if pd.notna(row[cols["zawod"]]) else "",
            umiejetnosci_obowiazki=str(row[cols["obowiazki"]]) if pd.notna(row[cols["obowiazki"]]) else "",
            model=model,
            title_emb=title_emb,
            tasks_emb=tasks_emb,
            synteza_emb=synteza_emb,
            codes_ordered=codes_ordered,
            metadata=metadata,
            prefix=prefix,
        )
    candidates = st.session_state[cache_key]

    preview_code = (prefix or "") + "0"
    NO_DETERMINATION_OPTION = (
        f"0 — Brak możliwości ustalenia dokładnej cyfry (dopełnij pozostałe cyfry zerami, kod: {preview_code})"
    )
    show_no_determination = level > 1

    # Na 1. kroku kaskady (grupa główna) koder może zamiast tego stwierdzić, że
    # w ogóle nie jest w stanie zakodować danej osoby - wybranie tej opcji
    # kończy kodowanie tej osoby od razu (bez wypełniania cyfr zerami) i
    # przechodzi do kolejnej osoby.
    NO_CODE_OPTION = "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)"
    show_no_code = level == 1

    options = [
        _format_candidate_label(r.isco_code, r.title_pl, getattr(r, "title_en", ""), r.score)
        for r in candidates.itertuples()
    ]
    if show_no_determination:
        options = options + [NO_DETERMINATION_OPTION]
    if show_no_code:
        options = options + [NO_CODE_OPTION]

    if not options:
        st.warning(
            "Brak kodów ISCO-08 pasujących do wybranego dotychczas prefiksu. "
            "Cofnij się o krok i wybierz inną cyfrę."
        )
        selected_vars = []
        ai_helpfulness = None
        choice = None
        is_no_determination = False
        is_uncodable = False
        chosen_code = None
    else:
        choice = st.radio(
            "Wybierz kod pasujący do tego kroku",
            options=options,
            index=None,
            key=f"cascade_choice_{idx}_{level}_{prefix}",
            on_change=_mark_first_interaction,
            args=(idx,),
        )
        is_no_determination = choice == NO_DETERMINATION_OPTION
        is_uncodable = show_no_code and choice == NO_CODE_OPTION

        if choice is None:
            chosen_code = None
            chosen_title = None
        elif is_no_determination or is_uncodable:
            chosen_code = None
            chosen_title = None
        else:
            choice_idx = options.index(choice)
            chosen_code = candidates.iloc[choice_idx]["isco_code"]
            chosen_title = candidates.iloc[choice_idx]["title_pl"]

        # Zmienne, z których korzystał koder = dowolna kombinacja kolumn
        # zaznaczonych w widocznej tabeli "Dane respondenta". Nie filtrujemy
        # po dtype: kategorie ESS bywają wczytane jako tekst (np. B31), mimo
        # że są pełnoprawnymi zmiennymi klasyfikacyjnymi.
        source_cols = set(visible_df_for_mode(df.iloc[[idx]], target).columns)
        var_meta = load_var_metadata(target)
        table_selection = st.session_state.get(_resp_table_key(idx, df_state_key), {})
        clicked_cols = table_selection.get("selection", {}).get("columns", [])
        selected_vars = [c for c in clicked_cols if c in source_cols]

        _render_selected_vars_caption(row, var_meta, selected_vars)
        _render_previously_used_vars_caption(df, idx, selected_vars)

        # Ocena AI pojawia się tylko na ostatnim FAKTYCZNIE osiągniętym
        # poziomie szczegółowości: albo poziom 4, albo moment wyboru "brak
        # możliwości ustalenia dokładnej cyfry" (koniec kodowania). Pole
        # uzasadnienia jest jedno, wspólne dla obu ścieżek zapisu (wybór z
        # listy i wpisanie kodu ręcznie) - patrz `direct_uzasadnienie` niżej,
        # więc nie duplikujemy go tutaj.
        show_uzasadnienie = level == 4 or is_no_determination or is_uncodable
        if show_uzasadnienie:
            ai_helpfulness = st.slider(
                "Jak pomocne były dopasowane kody podczas kodowania hierarchicznego?",
                min_value=1,
                max_value=5,
                value=3,
                key=f"cascade_ai_helpfulness_{idx}_{level}_{prefix}",
            )
        else:
            ai_helpfulness = None

    st.markdown("**Lub wpisz od razu pełny, 4-cyfrowy kod ISCO-08:**")
    _seed_direct_code_and_uzasadnienie(
        df, idx,
        direct_key=f"cascade_direct_code_{idx}",
        uzasadnienie_key=f"cascade_direct_uzasadnienie_{idx}",
    )
    direct_code = st.text_input(
        "Pełny kod ISCO-08",
        max_chars=4,
        placeholder="",
        key=f"cascade_direct_code_{idx}",
        label_visibility="collapsed",
    ).strip()
    direct_uzasadnienie = st.text_area(
        "Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)",
        key=f"cascade_direct_uzasadnienie_{idx}",
        height=70,
    )
    valid_final_codes = set(load_embeddings_level(4)[3])
    valid_level1_codes = set(load_embeddings_level(1)[3])
    valid_level2_codes = set(load_embeddings_level(2)[3])
    valid_level3_codes = set(load_embeddings_level(3)[3])
    if st.button(
        "Zapisz pełny kod i przejdź dalej",
        type="primary",
        use_container_width=True,
        key=f"cascade_direct_save_{idx}",
    ):
        if not (len(direct_code) == 4 and direct_code.isdigit()):
            st.warning("Wpisz 4 cyfry kodu ISCO-08.")
        elif not _direct_code_is_valid(direct_code, valid_final_codes, valid_level1_codes, valid_level2_codes, valid_level3_codes):
            st.warning(
                "Podany kod nie występuje na liście kodów ISCO-08 i nie jest poprawnym "
                "prefiksem dopełnionym zerami (np. 5200)."
            )
        else:
            _cascade_save_direct_code(
                df, idx, direct_code, target, direct_uzasadnienie,
                df_state_key, idx_state_key, qualifying_positions,
            )
            st.rerun()

    st.divider()

    col_back, col_next, col_cancel = st.columns(3)

    with col_back:
        if level > 1 and st.button("← Cofnij krok", use_container_width=True, key=f"cascade_back_{idx}"):
            digits.pop()
            st.session_state[f"cascade_step_{idx}"] = level - 1
            st.session_state[f"hitl_wracal_{idx}"] = True
            st.rerun()

    with col_cancel:
        if st.button("Anuluj kodowanie kaskadowe", use_container_width=True, key=f"cascade_cancel_{idx}"):
            _cancel_cascade(idx)
            st.rerun()

    with col_next:
        if is_uncodable:
            next_label = "Zapisz (brak możliwości zakodowania) i przejdź dalej"
        elif level == 4 or is_no_determination:
            next_label = "Zatwierdź kod finalny"
        else:
            next_label = "Dalej →"
        if options and st.button(next_label, type="primary", use_container_width=True, key=f"cascade_next_{idx}"):
            if choice is None:
                st.warning("Wybierz jedną opcję przed przejściem dalej.")
            else:
                df.at[idx, f"ISCO_poziom{level}_zmienne"] = ", ".join(selected_vars) if selected_vars else None
                rank, score = _get_rank_and_score(candidates, chosen_code)
                df.at[idx, f"ISCO_poziom{level}_ranking_pozycja"] = rank
                df.at[idx, f"ISCO_poziom{level}_score"] = score
                if show_uzasadnienie and direct_uzasadnienie.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = direct_uzasadnienie.strip()

                # Zaznaczenie kolumn w tabeli "Dane respondenta" NIE resetuje się
                # przy przejściu na kolejny poziom kaskady (_resp_table_key nie
                # zależy już od cascade_step_{idx}) - koder widzi te same
                # zaznaczone zmienne na każdym kolejnym kroku, aż do zmiany osoby.

                if is_uncodable:
                    # Koder jednoznacznie stwierdził, że nie da się zakodować tej
                    # osoby do żadnego kodu ISCO-08 - nie wypełniamy cyfr zerami
                    # (to celowo inne od "brak możliwości ustalenia" na dalszych
                    # krokach), tylko oznaczamy przypadek i przechodzimy dalej.
                    df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                    df.at[idx, "Kodowany_podmiot"] = target
                    _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5", df_state_key=df_state_key)
                    _persist_df(df_state_key, df)
                    _cancel_cascade(idx)
                    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                    st.rerun()
                elif is_no_determination:
                    fill_count = 4 - len(digits)
                    digits.extend(["0"] * fill_count)
                    final_code = "".join(digits)
                    df.at[idx, "ISCO_poziom1"] = digits[0]
                    df.at[idx, "ISCO_poziom2"] = digits[1]
                    df.at[idx, "ISCO_poziom3"] = digits[2]
                    df.at[idx, "ISCO_poziom4"] = digits[3]
                    df.at[idx, "ISCO_PRED"] = final_code
                    df.at[idx, "ISCO_wybrany"] = final_code
                    df.at[idx, "Kodowany_podmiot"] = target
                    _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5", df_state_key=df_state_key)
                    _persist_df(df_state_key, df)
                    _cancel_cascade(idx)
                    _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                    st.rerun()
                else:
                    new_digit = chosen_code[-1]
                    digits.append(new_digit)

                    if level == 4:
                        final_code = "".join(digits)
                        df.at[idx, "ISCO_poziom1"] = digits[0]
                        df.at[idx, "ISCO_poziom2"] = digits[1]
                        df.at[idx, "ISCO_poziom3"] = digits[2]
                        df.at[idx, "ISCO_poziom4"] = digits[3]
                        df.at[idx, "ISCO_PRED"] = final_code
                        df.at[idx, "ISCO_wybrany"] = final_code
                        df.at[idx, "Kodowany_podmiot"] = target
                        _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5", df_state_key=df_state_key)
                        _persist_df(df_state_key, df)
                        _cancel_cascade(idx)
                        _set_idx(idx_state_key, _next_idx_after_save(qualifying_positions, idx, idx_state_key, n))
                        st.rerun()
                    else:
                        df.at[idx, f"ISCO_poziom{level}"] = new_digit
                        _persist_df(df_state_key, df)
                        st.session_state[f"cascade_step_{idx}"] = level + 1
                        st.rerun()

    st.caption("Skrót: Ctrl+Enter (na macOS także Cmd+Enter) uruchamia główny przycisk bieżącego kroku.")
    _ctrl_enter_shortcut(
        handler_key="__cascadeCtrlEnterHandler",
        fallback_labels=[
            "Zapisz (brak możliwości zakodowania) i przejdź dalej",
            "Zatwierdź kod finalny",
            "Dalej →",
        ],
        direct_input_aria_label="Pełny kod ISCO-08",
        direct_save_label="Zapisz pełny kod i przejdź dalej",
    )
    _instant_text_persistence(idx, [
        ("Pełny kod ISCO-08", direct_code),
        ("Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)", direct_uzasadnienie),
    ])


@st.dialog("Szczegóły zmiennej")
def _show_variable_dialog(col: str, raw_val, label: str, value_labels: dict):
    """Modal (z natywnym X do zamknięcia) pokazujący pełny opis zmiennej:
    nazwa, etykieta, i każda kategoria w osobnej linii - pogrubiona ta,
    która odpowiada faktycznej wartości respondenta."""
    st.markdown(f"### {col}")
    if label:
        st.caption(label)
    st.markdown(f"**Wartość respondenta:** `{raw_val}`")

    matched_key = None
    if pd.notna(raw_val) and value_labels:
        key_candidates = [str(raw_val)]
        try:
            key_candidates.append(str(int(float(raw_val))))
        except (ValueError, TypeError):
            pass
        for k in key_candidates:
            if k in value_labels:
                matched_key = k
                break

    st.divider()

    if not value_labels:
        st.caption("Brak zdefiniowanych kategorii dla tej zmiennej.")
    else:
        for k, v in value_labels.items():
            if k == matched_key:
                st.markdown(f"➡️ **{k} = {v}**")
            else:
                st.markdown(f"{k} = {v}")


def _direct_code_is_valid(code: str, valid_final_codes, valid_level1_codes, valid_level2_codes, valid_level3_codes) -> bool:
    """Akceptuje pełny, konkretny kod ISCO-08 (poziom 4) ORAZ kod
    niedoprecyzowany - prefiks o długości 1-3 cyfr dopełniony zerami do
    4 cyfr (np. 5200 = grupa 52, nieustalona dokładna cyfra), analogicznie
    do opcji "Brak możliwości ustalenia dokładnej cyfry" w kodowaniu
    kaskadowym. Wspólna dla Metody A i pól "wpisz od razu pełny kod" w
    module B/C, żeby zasady walidacji były wszędzie identyczne."""
    if code in valid_final_codes:
        return True
    if code.endswith("000") and code[:1] in valid_level1_codes:
        return True
    if code.endswith("00") and not code.endswith("000") and code[:2] in valid_level2_codes:
        return True
    if code.endswith("0") and not code.endswith("00") and code[:3] in valid_level3_codes:
        return True
    return False


def _previously_used_vars(df, idx: int) -> list[str]:
    """Odtwarza listę zmiennych (kolumn), z których koder korzystał przy
    ostatniej decyzji dla tego przypadku - na podstawie zapisanych kolumn
    ISCO_poziomX_zmienne. Używane WYŁĄCZNIE do wyświetlenia (read-only
    caption) - nie da się nią przywrócić samego zaznaczenia w tabeli
    (Streamlit blokuje programową zmianę stanu widgetu st.dataframe z
    on_select), ale koder może chociaż zobaczyć, czego używał poprzednio."""
    cols: list[str] = []
    seen = set()
    for level in (1, 2, 3, 4):
        col = f"ISCO_poziom{level}_zmienne"
        if col not in df.columns:
            continue
        val = df.at[idx, col]
        if isinstance(val, str) and val.strip():
            for v in val.split(","):
                v = v.strip()
                if v and v not in seen:
                    seen.add(v)
                    cols.append(v)
    return cols


def _render_previously_used_vars_caption(df, idx: int, live_selected_vars: list[str]) -> None:
    """Pokazuje (jeśli nic nie jest aktualnie zaznaczone w tabeli) listę
    zmiennych użytych przy poprzedniej decyzji dla tego przypadku - żeby
    koder wracający do już zakodowanego przypadku widział, czego wcześniej
    użył, nawet gdy samo zaznaczenie w tabeli się nie przywróciło."""
    if live_selected_vars:
        return
    previous = _previously_used_vars(df, idx)
    if previous:
        st.caption("Poprzednio użyte zmienne (z ostatniej decyzji dla tego przypadku): " + ", ".join(previous))


def _saved_answer(df, idx: int) -> dict:
    """Zwraca poprzednio zapisaną decyzję dla wiersza (kod + uzasadnienie),
    jeśli istnieje - używane do przywrócenia widoku (wybór na liście top-10,
    pole 'Uzasadnienie') po powrocie do już zakodowanego przypadku."""
    code = df.at[idx, "ISCO_wybrany"] if "ISCO_wybrany" in df.columns else None
    uzasadnienie = df.at[idx, "Uzasadnienie_finalne"] if "Uzasadnienie_finalne" in df.columns else None
    return {
        "code": code if isinstance(code, str) and code.strip() else None,
        "uzasadnienie": uzasadnienie if isinstance(uzasadnienie, str) and uzasadnienie.strip() else "",
    }


def _seed_top10_widgets(df, idx: int, options: list[str], radio_key: str, direct_key: str, uzasadnienie_key: str) -> None:
    """Przywraca stan widgetów kroku top-10 (wybór z listy / kod wpisany
    ręcznie / uzasadnienie) na podstawie ostatnio zapisanej decyzji dla tego
    przypadku - ale TYLKO jeśli żaden z widgetów nie ma jeszcze stanu w
    bieżącej sesji (żeby nie nadpisywać tego, co koder właśnie klika)."""
    saved = _saved_answer(df, idx)
    if radio_key not in st.session_state and saved["code"]:
        # Etykiety opcji mają format "**<kod>** — ..." (patrz _format_candidate_label) -
        # dopasowanie musi uwzględniać markdown pogrubienia, inaczej nigdy nie trafi.
        matched = next((opt for opt in options if opt.startswith(f"**{saved['code']}**")), None)
        if matched:
            st.session_state[radio_key] = matched
        elif direct_key not in st.session_state:
            # Zapisany kod nie znalazł się wśród aktualnych propozycji top-10
            # (np. inna kolejność rankingu) - pokazujemy go w polu na kod wpisany
            # ręcznie, żeby koder od razu widział swoją poprzednią decyzję.
            st.session_state[direct_key] = saved["code"]
    if uzasadnienie_key not in st.session_state and saved["uzasadnienie"]:
        st.session_state[uzasadnienie_key] = saved["uzasadnienie"]


def _seed_direct_code_and_uzasadnienie(df, idx: int, direct_key: str, uzasadnienie_key: str) -> None:
    """Jak _seed_top10_widgets, ale dla ekranów bez listy top-10 (Metoda A,
    kaskada) - przywraca pole 'pełny kod ISCO-08' i pole uzasadnienia na
    podstawie ostatnio zapisanej decyzji, jeśli widgety jeszcze nie mają
    stanu w bieżącej sesji."""
    saved = _saved_answer(df, idx)
    if direct_key not in st.session_state and saved["code"]:
        st.session_state[direct_key] = saved["code"]
    if uzasadnienie_key not in st.session_state and saved["uzasadnienie"]:
        st.session_state[uzasadnienie_key] = saved["uzasadnienie"]


# Prefiksy kluczy widgetów, których BIEŻĄCY (jeszcze niezapisany) stan wolno
# zrzucić na dysk i odtworzyć po odświeżeniu strony (patrz
# _persist_widget_drafts / _restore_widget_drafts). Świadomie NIE obejmuje:
# - przycisków (st.button) - odtworzenie ich stanu mogłoby fałszywie ponownie
#   wywołać akcję (np. zapis albo start kaskady) zaraz po odświeżeniu;
# - "resp_table_" (zaznaczenie kolumn w st.dataframe z on_select) - Streamlit
#   BLOKUJE programowe ustawianie stanu tego typu widgetu przez
#   st.session_state (StreamlitValueAssignmentNotAllowedError), więc nie da
#   się go w ten sposób przywrócić.
# "manual_step_"/"manual_digits_" i "cascade_step_"/"cascade_digits_" to
# ZWYKŁE zmienne w session_state (nie klucze widgetów), więc programowe
# ustawienie ich przez session_state jest bezpieczne - dzięki temu poziom
# kaskady (i dotychczas wybrane cyfry), na którym koder akurat jest, też
# przetrwa odświeżenie strony, zamiast zawsze wracać do poziomu 1.
DRAFT_KEY_PREFIXES = (
    "hitl_choice_", "hitl_direct_code_", "hitl_uzasadnienie_", "hitl_ai_helpfulness_",
    "pa_top10_choice_", "pa_top10_direct_code_", "pa_top10_uzasadnienie_", "pa_top10_ai_helpfulness_",
    "pa1_decision_", "pa1_komentarz_",
    "manual_direct_code_", "manual_uzasadnienie_", "manual_choice_",
    "manual_step_", "manual_digits_",
    "cascade_choice_", "cascade_direct_code_", "cascade_uzasadnienie_",
    "cascade_direct_uzasadnienie_", "cascade_ai_helpfulness_",
    "cascade_step_", "cascade_digits_",
)


def _is_json_safe(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _draft_cache_path(df_state_key: str) -> Path:
    username = st.session_state.get("username", "anon")
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_CACHE_DIR / f"{username}_{df_state_key}_draft.json"


def _persist_widget_drafts(df_state_key: str, idx: int) -> None:
    """Zapisuje na dysk BIEŻĄCY, jeszcze niezapisany stan widgetów dla
    przypadku `idx` - wybór z listy, wpisany tekst, zaznaczenia w tabeli -
    żeby przetrwał odświeżenie strony (F5) w trakcie wypełniania, ZANIM koder
    kliknie 'Zapisz'. Wywoływane na końcu renderowania strony, więc łapie
    stan wszystkich pasujących widgetów utworzonych w tym przebiegu."""
    id_pattern = re.compile(rf"(?:^|_){idx}(?:_|$)")
    draft = {}
    for key, value in st.session_state.items():
        if not isinstance(key, str) or not key.startswith(DRAFT_KEY_PREFIXES):
            continue
        if not id_pattern.search(key):
            continue
        if _is_json_safe(value):
            draft[key] = value
    try:
        _draft_cache_path(df_state_key).write_text(
            json.dumps({"idx": idx, "values": draft}), encoding="utf-8"
        )
    except OSError:
        pass


def _restore_widget_drafts(df_state_key: str, idx: int) -> None:
    """Odtwarza zapisany szkic (patrz _persist_widget_drafts) - TYLKO jeśli
    dotyczy dokładnie tego samego przypadku `idx` (inaczej porzuca stary
    szkic, żeby nie podstawić cudzych odpowiedzi pod inny przypadek) i tylko
    dla kluczy, które nie mają jeszcze wartości w bieżącej sesji (czyli to
    świeży start po odświeżeniu, a nie nadpisywanie czegoś, co koder właśnie
    kliknął)."""
    path = _draft_cache_path(df_state_key)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("idx") != idx:
        return
    for key, value in data.get("values", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def _resp_table_key(idx: int, df_state_key: str) -> str:
    """Klucz widgetu tabeli 'Dane respondenta' (st.dataframe z zaznaczaniem kolumn).

    Klucz NIE zależy od aktualnego kroku kaskady (cascade_step_{idx}) - dzięki
    temu zaznaczone kolumny (zmienne) pozostają zaznaczone przy przejściu do
    kolejnego lub poprzedniego kroku kaskady, zamiast odznaczać się na każdym
    nowym ekranie. Widget zachowuje stan zaznaczenia, dopóki koder nie przejdzie
    do innego respondenta (idx) lub nie zmieni się namespace modułu.

    Namespace modułu zapobiega współdzieleniu stanu tabeli pomiędzy trybem
    2 (hitl1d_df) i 3 (hitl_df) dla tego samego numeru respondenta."""
    return f"resp_table_{df_state_key}_{idx}"


def _handle_table_column_click(df, idx: int, row, var_meta: dict, selection_key: str):
    """Odczytuje zaznaczenie kolumny z tabeli (kliknięcie nagłówka) i otwiera
    modal ze szczegółami tej zmiennej - tylko raz na nowe zaznaczenie, żeby
    modal nie odnawiał się w kółko przy każdym kolejnym rerunie."""
    event = st.session_state.get(selection_key)
    selected_cols = []
    if event is not None:
        selected_cols = event.get("selection", {}).get("columns", [])

    last_key = f"{selection_key}_last"
    if selected_cols:
        col = selected_cols[0]
        if st.session_state.get(last_key) != col and var_meta.get(col, {}).get("label"):
            st.session_state[last_key] = col
            meta = var_meta.get(col, {})
            _show_variable_dialog(
                col,
                row.get(col),
                meta.get("label", ""),
                meta.get("value_labels", {}) or {},
            )
    else:
        st.session_state[last_key] = None


# ============================================================
# STRONA: KLASYFIKACJA Z UDZIAŁEM EKSPERTA (moduł C)
# ============================================================
def render_classify_hitl():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    render_logo_header()

    if st.button("← Wróć do strony głównej", key="back_hitl"):
        go_to("home")
        st.rerun()

    st.markdown(
        '<h1 style="text-align:center; line-height:1.3;">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta</h1>',
        unsafe_allow_html=True,
    )
    st.write(
        "Wczytaj plik CSV zawierający dane ankietowe European Social Survey (ESS). "
        "Dla każdego respondenta lub jego partnera system wyświetla listę najbardziej "
        "prawdopodobnych kodów ISCO-08 wygenerowanych przez model. Jeżeli żaden z "
        "proponowanych kodów nie jest właściwy, możliwe jest przeprowadzenie klasyfikacji "
        "kaskadowej, polegającej na wyborze kodu ISCO-08 krok po kroku, od najwyższego do "
        "najniższego poziomu szczegółowości. Zadaniem eksperta jest wskazanie najbardziej "
        "odpowiedniego kodu zawodu, a wszystkie podjęte decyzje są automatycznie zapisywane "
        "przez system."
    )

    uploaded_file = st.file_uploader("Wybierz plik CSV", type=["csv"], key="uploader_hitl")

    if uploaded_file is None:
        # Po odświeżeniu strony uploaded_file zawsze wraca jako None - próbujemy
        # najpierw odtworzyć df z cache na dysku (patrz _load_df_cache), zanim
        # skasujemy postęp kodowania.
        if "hitl_df" not in st.session_state:
            cached_df, cached_source = _load_df_cache("hitl_df")
            if cached_df is not None:
                st.session_state["hitl_df"] = cached_df
                st.session_state["hitl_source"] = cached_source
                resume_target_main = _get_coding_target("hitl_df")
                fallback_idx = _first_unfinished_idx(cached_df, resume_target_main)
                st.session_state[_frontier_key("hitl_idx")] = _restore_frontier_from_query("hitl_idx", fallback_idx)
                _set_idx("hitl_idx", _restore_idx_from_query("hitl_idx", fallback_idx, len(cached_df)))
        if "hitl_df" not in st.session_state:
            return

    # Wczytanie pliku tylko raz (przy zmianie pliku resetujemy stan)
    if uploaded_file is not None and (
        "hitl_df" not in st.session_state or st.session_state.get("hitl_source") != uploaded_file.name
    ):
        df = read_csv_robust(uploaded_file)
        for col in ("B33", "B34", "B35", "B48", "B49", "B50"):
            if col not in df.columns:
                st.error(f"W pliku nie znaleziono wymaganej kolumny: {col}")
                return

        if "ISCO_wybrany" not in df.columns:
            df["ISCO_wybrany"] = None
            df["Decyzja_kodera_zawod"] = None
            df["Decyzja_kodera_notatka"] = None
        if "Kodowany_podmiot" not in df.columns:
            df["Kodowany_podmiot"] = None
        if "Brak_mozliwosci_zakodowania" not in df.columns:
            df["Brak_mozliwosci_zakodowania"] = None
        for col in ("ISCO_poziom1", "ISCO_poziom2", "ISCO_poziom3", "ISCO_poziom4", "ISCO_PRED"):
            if col not in df.columns:
                df[col] = None
        for col in (
            "ISCO_poziom1_zmienne",
            "ISCO_poziom2_zmienne",
            "ISCO_poziom3_zmienne",
            "ISCO_poziom4_zmienne",
            "ISCO_poziom1_ranking_pozycja",
            "ISCO_poziom1_score",
            "ISCO_poziom2_ranking_pozycja",
            "ISCO_poziom2_score",
            "ISCO_poziom3_ranking_pozycja",
            "ISCO_poziom3_score",
            "ISCO_poziom4_ranking_pozycja",
            "ISCO_poziom4_score",
            "Ranking_pozycja_wybranego_kodu",
            "Score_wybranego_kodu",
            "Uzasadnienie_finalne",
            "Czas_kodowania_sekundy",
            "Czas_do_pierwszej_interakcji_sekundy",
            "Czy_uzytkownik_wracal",
            "Ocena_AI_top10_1_5",
            "Ocena_AI_kaskadowo_1_5",
        ):
            if col not in df.columns:
                df[col] = None

        # Wymuszenie właściwego dtype (patrz _ensure_text_column_dtype) - kluczowe
        # przy WZNOWIENIU kodowania z częściowo wypełnionego pliku CSV, w którym
        # pandas mógł błędnie nadać kolumnom z kodami ISCO-08 typ float64.
        for col in (
            "ISCO_wybrany",
            "Decyzja_kodera_zawod",
            "Decyzja_kodera_notatka",
            "Kodowany_podmiot",
            "Brak_mozliwosci_zakodowania",
            "ISCO_poziom1",
            "ISCO_poziom2",
            "ISCO_poziom3",
            "ISCO_poziom4",
            "ISCO_PRED",
            "ISCO_poziom1_zmienne",
            "ISCO_poziom2_zmienne",
            "ISCO_poziom3_zmienne",
            "ISCO_poziom4_zmienne",
            "Uzasadnienie_finalne",
        ):
            _ensure_text_column_dtype(df, col)
        _ensure_object_dtype(df, "Czy_uzytkownik_wracal")

        resume_target_main = _get_coding_target("hitl_df")
        resume_idx = _first_unfinished_idx(df, resume_target_main)
        _set_idx("hitl_idx", resume_idx)
        st.session_state["hitl_source"] = uploaded_file.name
        _persist_df("hitl_df", df, source_name=uploaded_file.name)
        if 0 < resume_idx < len(df):
            _, resume_rank, resume_total = _qualifying_progress(
                _qualifying_positions(df, resume_target_main), resume_idx, len(df)
            )
            st.session_state["hitl_resume_msg"] = (
                f"Wykryto częściowo wypełniony plik - wznowiono kodowanie od osoby nr {resume_rank} z {resume_total}."
            )
        elif resume_idx >= len(df) and len(df) > 0:
            st.session_state["hitl_resume_msg"] = (
                "Wykryto plik, w którym wszystkie osoby w bieżącym trybie są już zakodowane."
            )

    df = st.session_state["hitl_df"]
    idx = st.session_state["hitl_idx"]
    n = len(df)
    if idx < n:
        _restore_widget_drafts("hitl_df", idx)

    resume_msg = st.session_state.pop("hitl_resume_msg", None)
    if resume_msg:
        st.info(resume_msg)
    st.write("")
    render_mode_selector("hitl_df", "hitl_idx", df, idx, n)
    mode_main = _get_coding_target("hitl_df")
    _warn_if_meta_missing(mode_main)
    st.write("")
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        podmiot_label = "Partner" if mode_main == "Partner" else "Respondent"
        qualifying_positions_main = _qualifying_positions(df, mode_main)
        progress_fraction, progress_rank, progress_total = _qualifying_progress(qualifying_positions_main, idx, n)
        st.progress(progress_fraction, text=f"{podmiot_label} {progress_rank} z {progress_total}")
    with col_nav2:
        with st.popover("Podgląd danych"):
            visible_df = visible_df_for_mode(df, mode_main)
            st.dataframe(visible_df, use_container_width=True, column_config=build_column_config(visible_df, load_var_metadata(mode_main)))

    _render_case_jumper(qualifying_positions_main, idx, "hitl_idx", key_suffix=f"hitl_{mode_main}")

    previous_hitl_idx = _prev_qualifying_idx(qualifying_positions_main, idx)
    prev_label_top = "← Poprzedni partner" if mode_main == "Partner" else "← Poprzedni respondent"
    col_prev_top, col_next_top = st.columns(2)
    with col_prev_top:
        if previous_hitl_idx != idx:
            if st.button(prev_label_top, key=f"hitl_prev_case_top_{idx}", use_container_width=True):
                st.session_state[f"hitl_wracal_{previous_hitl_idx}"] = True
                _set_idx("hitl_idx", previous_hitl_idx)
                st.rerun()
    with col_next_top:
        if _render_next_case_button(qualifying_positions_main, idx, "hitl_idx", key_suffix=f"hitl_top_{mode_main}"):
            st.rerun()

    # Kolejność kolumn w eksporcie - ta sama zarówno dla wyniku KOŃCOWEGO
    # (po ukończeniu kodowania), jak i dla podglądu/pobrania CZĘŚCIOWEGO
    # postępu w dowolnym momencie (patrz przycisk niżej i sekcja "if idx >= n").
    ref_cols = [c for c in ["B33", "B34", "B35", "B48", "B49", "B50", "ISCO08"] if c in df.columns]
    result_cols = [
        "Kodowany_podmiot",
        "Brak_mozliwosci_zakodowania",
        "ISCO_wybrany",
        "ISCO_poziom1",
        "ISCO_poziom1_zmienne",
        "ISCO_poziom1_ranking_pozycja",
        "ISCO_poziom1_score",
        "ISCO_poziom2",
        "ISCO_poziom2_zmienne",
        "ISCO_poziom2_ranking_pozycja",
        "ISCO_poziom2_score",
        "ISCO_poziom3",
        "ISCO_poziom3_zmienne",
        "ISCO_poziom3_ranking_pozycja",
        "ISCO_poziom3_score",
        "ISCO_poziom4",
        "ISCO_poziom4_zmienne",
        "ISCO_poziom4_ranking_pozycja",
        "ISCO_poziom4_score",
        "ISCO_PRED",
        "Ranking_pozycja_wybranego_kodu",
        "Score_wybranego_kodu",
        "Uzasadnienie_finalne",
        "Decyzja_kodera_zawod",
        "Decyzja_kodera_notatka",
        "Czas_kodowania_sekundy",
        "Czas_do_pierwszej_interakcji_sekundy",
        "Czy_uzytkownik_wracal",
        "Ocena_AI_top10_1_5",
        "Ocena_AI_kaskadowo_1_5",
    ]
    other_cols = [c for c in df.columns if c not in ref_cols and c not in result_cols]
    export_df = df[other_cols + ref_cols + result_cols].copy()
    mode_suffix = "respondent" if mode_main == "Respondent" else "partner"

    # Pobranie CZĘŚCIOWEGO wyniku - dostępne cały czas w trakcie kodowania
    # (nie trzeba czekać na ukończenie całego pliku). Format pliku jest
    # identyczny jak wynik końcowy, więc częściowy plik można bez przeszkód
    # wgrać z powrotem później - aplikacja sama wznowi kodowanie od pierwszej
    # nieukończonej osoby (patrz _first_unfinished_idx).
    if idx < n:
        with st.expander(f"Pobierz częściowy wynik (dotychczasowy postęp: {progress_rank - 1} z {progress_total})"):
            unfinished_main = _unfinished_case_numbers(df, mode_main, qualifying_positions_main)
            if unfinished_main:
                st.caption(
                    f"Nieukończone numery przypadków ({len(unfinished_main)}): "
                    + ", ".join(str(n) for n in unfinished_main)
                )
            partial_csv, partial_xlsx = _build_csv_xlsx_bytes(export_df)
            col_pdl1, col_pdl2 = st.columns(2)
            with col_pdl1:
                st.download_button(
                    "Pobierz częściowy wynik (CSV)",
                    data=partial_csv,
                    file_name=f"wynik_czesciowy_klasyfikacji_ekspert_{mode_suffix}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="hitl_partial_dl_csv",
                )
            with col_pdl2:
                st.download_button(
                    "Pobierz częściowy wynik (Excel .xlsx)",
                    data=partial_xlsx,
                    file_name=f"wynik_czesciowy_klasyfikacji_ekspert_{mode_suffix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="hitl_partial_dl_xlsx",
                )

    if idx >= n:
        podmiot_plural = "partnerów" if mode_main == "Partner" else "respondentów"
        st.success(f"Sklasyfikowano wszystkich {podmiot_plural}.")

        csv_bytes, xlsx_bytes = _build_csv_xlsx_bytes(export_df)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Pobierz wynik (CSV)",
                data=csv_bytes,
                file_name=f"wynik_klasyfikacji_ekspert_{mode_suffix}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Pobierz wynik (Excel .xlsx)",
                data=xlsx_bytes,
                file_name=f"wynik_klasyfikacji_ekspert_{mode_suffix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        return

    row = df.iloc[idx]

    # Śledzenie czasu kodowania i powrotów dla tego respondenta
    st.session_state.setdefault(f"hitl_start_time_{idx}", time.time())
    st.session_state.setdefault(f"hitl_wracal_{idx}", False)

    _display_respondent_idno(row)
    st.write("Dane respondenta:")
    st.caption("Kliknij nazwy kolumn (zmiennych), z których korzystasz przy klasyfikacji.")

    _var_meta_debug = load_var_metadata(mode_main)

    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_main),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, _var_meta_debug),
        on_select="rerun",
        selection_mode=["multi-column"],
        key=_resp_table_key(idx, "hitl_df"),
    )

    cols = _target_cols("hitl_df")

    with st.container(border=True):
        st.markdown(f"**Zawód:** {row[cols['zawod']]}")
        st.markdown(f"**Obowiązki i zadania:** {row[cols['obowiazki']]}")
        st.markdown(f"**Wykształcenie:** {row[cols['wyksztalcenie']]}")

    cascade_active = f"cascade_step_{idx}" in st.session_state

    if cascade_active:
        render_cascade_step(df, idx, row)
        _persist_widget_drafts("hitl_df", idx)
        return

    target = _get_coding_target("hitl_df")
    qualifying_positions = _qualifying_positions(df, target)

    model = load_model()
    title_emb, tasks_emb, synteza_emb, codes_ordered, metadata = load_embeddings()

    cache_key = f"hitl_candidates_{idx}_{target}"
    if cache_key not in st.session_state:
        ranking = classify(
            zawod_czlowieka=str(row[cols["zawod"]]) if pd.notna(row[cols["zawod"]]) else "",
            umiejetnosci_obowiazki=str(row[cols["obowiazki"]]) if pd.notna(row[cols["obowiazki"]]) else "",
            wyksztalcenie=str(row[cols["wyksztalcenie"]]) if pd.notna(row[cols["wyksztalcenie"]]) else "",
            model=model,
            title_emb=title_emb,
            tasks_emb=tasks_emb,
            synteza_emb=synteza_emb,
            codes_ordered=codes_ordered,
            metadata=metadata,
            top_k=10,
        )
        st.session_state[cache_key] = ranking

    ranking = st.session_state[cache_key]

    NO_CODE_OPTION = "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)"
    options = [_format_candidate_label(r.isco_code, r.title_pl, getattr(r, "title_en", ""), r.score) for r in ranking.itertuples()]
    options.append(NO_CODE_OPTION)

    _seed_top10_widgets(
        df, idx, options,
        radio_key=f"hitl_choice_{idx}",
        direct_key=f"hitl_direct_code_{idx}",
        uzasadnienie_key=f"hitl_uzasadnienie_{idx}",
    )

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        index=None,
        key=f"hitl_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None
    is_uncodable = choice == NO_CODE_OPTION

    if choice is None:
        chosen_code = None
        chosen_title = None
    elif is_uncodable:
        chosen_code = None
        chosen_title = None
    else:
        choice_idx = options.index(choice)
        chosen_code = ranking.iloc[choice_idx]["isco_code"]
        chosen_title = ranking.iloc[choice_idx]["title_pl"]

    # Ocena pomocności listy 10 dopasowanych kodów - wymagana zawsze, niezależnie
    # od tego, czy koder wybierze jeden z nich, czy przejdzie do kodowania
    # kaskadowego (patrz przycisk "Zakoduj od zera" niżej).
    ai_helpfulness = render_helpfulness_scale(
        "Jak pomocne były dopasowane kody? (1-5 punktów)",
        key=f"hitl_ai_helpfulness_{idx}",
    )

    # Opcjonalna notatka do wyboru - dostępna zawsze, niezależnie od tego, czy
    # koder wybrał jeden z 10 dopasowanych kodów, czy "brak poprawnego kodu"
    # (na wzór pola "Uzasadnienie / komentarz" z trybu kaskadowego).
    st.markdown("**Lub wpisz od razu pełny, 4-cyfrowy kod ISCO-08:**")
    direct_code = st.text_input(
        "Pełny kod ISCO-08",
        max_chars=4,
        placeholder="",
        key=f"hitl_direct_code_{idx}",
        label_visibility="collapsed",
    ).strip()

    uzasadnienie_top10 = st.text_area(
        "Uzasadnienie / komentarz do wyboru (opcjonalnie)",
        key=f"hitl_uzasadnienie_{idx}",
        height=70,
    )

    valid_final_codes = set(load_embeddings_level(4)[3])
    valid_level1_codes = set(load_embeddings_level(1)[3])
    valid_level2_codes = set(load_embeddings_level(2)[3])
    valid_level3_codes = set(load_embeddings_level(3)[3])

    col_btn1 = st.container()
    with col_btn1:
        if st.button("Zapisz i przejdź dalej", type="primary", use_container_width=True):
            if direct_code:
                # Ręcznie wpisany kod ma pierwszeństwo przed zaznaczeniem na liście
                # radio - koder mógł zaznaczyć jakąś opcję wcześniej, a potem
                # zmienić zdanie i wpisać kod bezpośrednio.
                if not (len(direct_code) == 4 and direct_code.isdigit()):
                    st.warning("Wpisz 4 cyfry kodu ISCO-08.")
                elif not _direct_code_is_valid(direct_code, valid_final_codes, valid_level1_codes, valid_level2_codes, valid_level3_codes):
                    st.warning(
                        "Podany kod nie występuje na liście kodów ISCO-08 i nie jest poprawnym "
                        "prefiksem dopełnionym zerami (np. 5200)."
                    )
                else:
                    df.at[idx, "ISCO_wybrany"] = direct_code
                    df.at[idx, "ISCO_poziom1"] = direct_code[0]
                    df.at[idx, "ISCO_poziom2"] = direct_code[1]
                    df.at[idx, "ISCO_poziom3"] = direct_code[2]
                    df.at[idx, "ISCO_poziom4"] = direct_code[3]
                    df.at[idx, "ISCO_PRED"] = direct_code
                    if uzasadnienie_top10.strip():
                        df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                    df.at[idx, "Brak_mozliwosci_zakodowania"] = None
                    df.at[idx, "Kodowany_podmiot"] = target
                    _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key="hitl_df")
                    _persist_df("hitl_df", df)
                    _set_idx("hitl_idx", _next_idx_after_save(qualifying_positions, idx, "hitl_idx", n))
                    st.rerun()
            elif is_uncodable:
                df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key="hitl_df")
                _persist_df("hitl_df", df)
                _set_idx("hitl_idx", _next_idx_after_save(qualifying_positions, idx, "hitl_idx", n))
                st.rerun()
            elif choice is None:
                st.warning("Wybierz jedną opcję z listy albo wpisz kod ręcznie przed zapisaniem.")
            else:
                df.at[idx, "ISCO_wybrany"] = chosen_code
                df.at[idx, "Decyzja_kodera_zawod"] = decyzja_kodera_zawod
                df.at[idx, "Decyzja_kodera_notatka"] = decyzja_kodera_notatka
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                rank, score = _get_rank_and_score(ranking, chosen_code)
                df.at[idx, "Ranking_pozycja_wybranego_kodu"] = rank
                df.at[idx, "Score_wybranego_kodu"] = score
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5", df_state_key="hitl_df")
                _persist_df("hitl_df", df)
                _set_idx("hitl_idx", _next_idx_after_save(qualifying_positions, idx, "hitl_idx", n))
                st.rerun()

    st.caption("Skrót: Ctrl+Enter (na macOS także Cmd+Enter) zapisuje wybór i przechodzi dalej.")
    _ctrl_enter_shortcut(
        handler_key="__hitlCtrlEnterHandler",
        fallback_labels=["Zapisz i przejdź dalej"],
    )
    _instant_text_persistence(idx, [
        ("Pełny kod ISCO-08", direct_code),
        ("Uzasadnienie / komentarz do wyboru (opcjonalnie)", uzasadnienie_top10),
    ])

    st.write("")
    _cascade_start_label = (
        "Popraw kodowanie kaskadowo (od ostatnio wybranej cyfry)"
        if isinstance(df.at[idx, "ISCO_poziom1"], str) and df.at[idx, "ISCO_poziom1"].strip()
        else "Zakoduj zawód od zera"
    )
    if st.button(
        _cascade_start_label,
        key=f"cascade_start_{idx}",
        use_container_width=True,
    ):
        # Zapisujemy ocenę pomocności listy 10 kodów, zanim koder przejdzie
        # do kodowania kaskadowego - inaczej ta ocena nigdy by się nie zapisała.
        df.at[idx, "Ocena_AI_top10_1_5"] = ai_helpfulness
        _persist_df("hitl_df", df)
        _start_cascade(idx, df=df)
        st.rerun()

    _persist_widget_drafts("hitl_df", idx)


# ============================================================
# ROUTER
# ============================================================
require_login()

with st.sidebar:
    st.caption(f"Zalogowano: {st.session_state.get('username', '')}")
    if st.button("Kwestionariusz badawczy", type="primary", use_container_width=True, key="sidebar_questionnaire"):
        go_to("questionnaire")
        st.rerun()
    if st.session_state.get("username") in ADMIN_USERS:
        if st.button("Wyniki ankiet", use_container_width=True, key="sidebar_questionnaire_results"):
            go_to("questionnaire_results")
            st.rerun()
        if st.button("Ewidencja kodowania", use_container_width=True, key="sidebar_ewidencja"):
            go_to("ewidencja")
            st.rerun()
    if st.button("Wyloguj", use_container_width=True):
        for key in ("authenticated", "username"):
            st.session_state.pop(key, None)
        st.query_params.pop("u", None)
        st.query_params.pop("p", None)
        st.session_state.page = "home"
        st.rerun()

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "classify_manual":
    render_classify_manual()
elif st.session_state.page == "classify_hitl":
    render_classify_hitl()
elif st.session_state.page == "classify_hitl_1digit":
    render_classify_hitl_1digit()
elif st.session_state.page == "questionnaire":
    render_questionnaire()
elif st.session_state.page == "questionnaire_results":
    render_questionnaire_results()
elif st.session_state.page == "ewidencja":
    render_ewidencja()
