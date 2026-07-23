import base64
import csv
import hmac
import io
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

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
st.set_page_config(
    page_title="Klasyfikacja zawodów ISCO-08",
    page_icon="🧭",
    layout="wide",  # zmienione z "centered" - więcej miejsca poziomego, żeby długie nazwy
    # zawodów w liście kandydatów rzadziej wymagały skracania (patrz MAX_LABEL_LINE_LEN)
)

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
ADMIN_USERS = {"User1"}
QUESTIONNAIRE_DB_PATH = Path(
    os.environ.get("QUESTIONNAIRE_DB_PATH", APP_DIR / "data" / "questionnaire.sqlite3")
)

ASSET_DIR = APP_DIR / "assets"
LOGO_PATHS = {
    "UMCS": ASSET_DIR / "logoc.png",
    "IFiS PAN": ASSET_DIR / "ifis.jpeg",
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
    gap: 3.2rem;
    flex-wrap: nowrap;
    padding: 0.2rem 0 0.9rem 0;
    margin-bottom: 0.8rem;
}}
.logo-header__item {{
    display: flex;
    justify-content: center;
    align-items: center;
    min-width: 0;
}}
.logo-header__item img {{
    display: block;
    max-width: 440px;
    max-height: 136px;
    object-fit: contain;
}}
.logo-header__item--umcs img {{
    max-width: 462px;
    max-height: 143px;
}}
.logo-header__item--ifis img {{
    max-height: 164px;
}}
.logo-header__item--ess img {{
    max-width: 1085px;
    max-height: 336px;
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
    .logo-header__item--ifis img {{
        max-height: 116px;
    }}
    .logo-header__item--umcs img {{
        max-height: 122px;
    }}
    .logo-header__item--ess img {{
        max-height: 286px;
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
    """Render the standard Streamlit radio scale, spread across the row via CSS."""
    with st.container(key=f"{key}_spread"):
        return st.radio(
            label,
            options=[1, 2, 3, 4, 5],
            index=2,
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
            st.rerun()
        else:
            st.error("Nieprawidłowy użytkownik lub hasło.")
    st.markdown("</div>", unsafe_allow_html=True)


def require_login():
    if not st.session_state.get("authenticated", False):
        render_login()
        st.stop()


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


def _build_column_help(col: str, var_meta: dict) -> Optional[str]:
    """Buduje tekst dymka (tooltip) dla nagłówka kolumny na podstawie metadanych
    zmiennej: etykieta zmiennej + (jeśli jest ich rozsądnie mało) lista etykiet
    wartości. Zwraca None, jeśli brak metadanych dla tej kolumny."""
    meta = var_meta.get(col)
    if not meta:
        return None

    label = meta.get("label", "")
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
    prefix = f"{isco_code} — "
    budget = MAX_LABEL_LINE_LEN - len(prefix) - len(suffix)
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
        st.session_state[idx_state_key] = _first_unfinished_idx(df, target_mode)
    else:
        st.session_state[idx_state_key] = 0


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
                    st.session_state[idx_state_key] = _first_unfinished_idx(df, "Respondent")
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
                    st.session_state[idx_state_key] = _first_unfinished_idx(df, "Partner")
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
        meta = var_meta.get(col)
        if not meta:
            continue
        label = meta.get("label", "")
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

    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
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

    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
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
    st.session_state.page = "home"


def go_to(page_name: str):
    st.session_state.page = page_name


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
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
                '<span class="module-card-sub">kodowanie ręczne</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_manual", use_container_width=True):
                go_to("classify_manual")

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Metoda B'
                '<span class="module-card-sub">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_hitl", use_container_width=True):
                go_to("classify_hitl")

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
    return options + [{"code": "__NO_MATCH__", "label": "Nic nie pasuje"}]


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


def _manual_save_code(
    df,
    idx: int,
    final_code: str,
    target: str,
    uzasadnienie: str,
    df_state_key: str,
    idx_state_key: str,
    qualifying_positions: list[int],
) -> None:
    """Zapisuje kompletną decyzję Metody A i przechodzi do następnego przypadku."""
    for level, digit in enumerate(final_code, start=1):
        df.at[idx, f"ISCO_poziom{level}"] = digit
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
    _save_respondent_meta(df, idx)
    st.session_state[df_state_key] = df
    st.session_state.pop(f"manual_step_{idx}", None)
    st.session_state.pop(f"manual_digits_{idx}", None)
    st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, len(df))


def _manual_ctrl_enter_shortcut(idx: int) -> None:
    """Łączy Ctrl/Cmd+Enter z głównym przyciskiem bieżącego kroku Metody A."""
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const handlerKey = '__manualCtrlEnterHandler';
        if (window.parent[handlerKey]) {{
            doc.removeEventListener('keydown', window.parent[handlerKey], true);
        }}
        window.parent[handlerKey] = (event) => {{
            if (!(event.ctrlKey || event.metaKey) || event.key !== 'Enter') return;
            const directInput = doc.querySelector('input[placeholder="np. 2512"]');
            const labels = directInput && directInput.value.trim()
                ? ['Zapisz pełny kod i przejdź dalej']
                : ['Zatwierdź kod finalny', 'Dalej →'];
            const buttons = [...doc.querySelectorAll('button')];
            const button = buttons.find((item) => labels.includes(item.innerText.trim()));
            if (button && !button.disabled) {{
                event.preventDefault();
                button.click();
            }}
        }};
        doc.addEventListener('keydown', window.parent[handlerKey], true);
        </script>
        """,
        height=0,
    )


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

    uzasadnienie = st.text_area(
        "Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)",
        key=f"manual_uzasadnienie_{idx}_{level}_{prefix}",
        height=70,
    )

    target = _get_coding_target(df_state_key)
    n = len(df)
    qualifying_positions = _qualifying_positions(df, target)

    st.markdown("**Lub wpisz od razu pełny, 4-cyfrowy kod ISCO-08:**")
    direct_code = st.text_input(
        "Pełny kod ISCO-08",
        value="",
        max_chars=4,
        placeholder="np. 2512",
        key=f"manual_direct_code_{idx}",
        label_visibility="collapsed",
    ).strip()
    valid_final_codes = set(load_embeddings_level(4)[3])
    if st.button(
        "Zapisz pełny kod i przejdź dalej",
        type="primary",
        use_container_width=True,
        key=f"manual_direct_save_{idx}",
    ):
        if not (len(direct_code) == 4 and direct_code.isdigit()):
            st.warning("Wpisz 4 cyfry kodu ISCO-08.")
        elif direct_code not in valid_final_codes:
            st.warning("Podany kod nie występuje na liście kodów ISCO-08.")
        else:
            _manual_save_code(
                df, idx, direct_code, target, uzasadnienie, df_state_key,
                idx_state_key, qualifying_positions,
            )
            st.rerun()

    st.caption("Skrót: Ctrl+Enter (na macOS także Cmd+Enter) uruchamia główny przycisk bieżącego kroku.")
    _manual_ctrl_enter_shortcut(idx)

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
            if chosen_code == "__NO_MATCH__":
                final_code = (prefix + "0" * (5 - level))[:4]
                _manual_save_code(
                    df, idx, final_code, target, uzasadnienie, df_state_key,
                    idx_state_key, qualifying_positions,
                )
                st.rerun()
            elif level < 4:
                digits.append(chosen_code[-1])
                df.at[idx, f"ISCO_poziom{level}"] = chosen_code[-1]
                st.session_state[df_state_key] = df
                st.session_state[f"manual_step_{idx}"] = level + 1
                st.rerun()
            else:
                final_code = chosen_code
                _manual_save_code(
                    df, idx, final_code, target, uzasadnienie, df_state_key,
                    idx_state_key, qualifying_positions,
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
        '<span style="font-size:0.6em;">kodowanie ręczne</span></h1>',
        unsafe_allow_html=True,
    )
    st.write(
        "Wczytaj plik CSV zawierający dane ankietowe European Social Survey (ESS). "
        "Ekspert wybiera kod ISCO-08 ręcznie, poziom po poziomie, z pełnych list "
        "kodów uporządkowanych rosnąco."
    )

    uploaded_file = st.file_uploader("Wybierz plik CSV", type=["csv"], key="uploader_manual")

    if uploaded_file is None:
        st.session_state.pop("manual_df", None)
        st.session_state.pop("manual_idx", None)
        return

    if "manual_df" not in st.session_state or st.session_state.get("manual_source") != uploaded_file.name:
        df = read_csv_robust(uploaded_file)
        for col in ("B33", "B34", "B35", "B48", "B49", "B50"):
            if col not in df.columns:
                st.error(f"W pliku nie znaleziono wymaganej kolumny: {col}")
                return

        _init_manual_result_columns(df)
        resume_target_manual = _get_coding_target("manual_df")
        resume_idx = _first_unfinished_idx(df, resume_target_manual)
        st.session_state["manual_df"] = df
        st.session_state["manual_idx"] = resume_idx
        st.session_state["manual_source"] = uploaded_file.name
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

    mode_suffix = "respondent" if mode_manual == "Respondent" else "partner"

    # Pozwala wrócić do ostatnio zakodowanego przypadku również po ukończeniu
    # całego pliku. Ponowny zapis po prostu nadpisuje poprzednią decyzję.
    previous_manual_idx = _prev_qualifying_idx(qualifying_positions_manual, idx)
    if previous_manual_idx != idx:
        if st.button("← Wróć do poprzedniego przypadku", key=f"manual_prev_case_{idx}"):
            _manual_reset_idx(previous_manual_idx)
            st.session_state["manual_idx"] = previous_manual_idx
            st.rerun()

    if idx < n:
        with st.expander(f"Pobierz częściowy wynik (dotychczasowy postęp: {progress_rank - 1} z {progress_total})"):
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
    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_manual),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, load_var_metadata(mode_manual)),
    )

    cols = _target_cols("manual_df")
    with st.container(border=True):
        st.markdown(f"**Zawód:** {row[cols['zawod']]}")
        st.markdown(f"**Obowiązki i zadania:** {row[cols['obowiazki']]}")
        st.markdown(f"**Wykształcenie:** {row[cols['wyksztalcenie']]}")

    render_manual_step(df, idx, row)


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
    if st.button(button_label, type="primary", use_container_width=True, key=f"pa1_next_{idx}"):
        df.at[idx, "Cyfra1_zatwierdzona_expert"] = "Tak" if is_confirm else "Nie"
        if not is_confirm and komentarz.strip():
            df.at[idx, "Powod_odrzucenia_cyfry"] = komentarz.strip()
        st.session_state[df_state_key] = df

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

    NO_MATCH_OPTION = "Brak poprawnego kodu (decyzja kodera)"
    NO_CODE_OPTION = "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)"
    fill_code = f"{prefix}000"
    NO_DETERMINATION_OPTION = (
        f"Brak możliwości ustalenia dokładnej cyfry (dopełnij pozostałe cyfry zerami, kod: {fill_code})"
    )
    options = [_format_candidate_label(r.isco_code, r.title_pl, getattr(r, "title_en", ""), r.score) for r in ranking.itertuples()]
    options.append(NO_DETERMINATION_OPTION)
    options.append(NO_MATCH_OPTION)
    options.append(NO_CODE_OPTION)

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        key=f"pa_top10_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None
    decyzja_kodera_kod = ""
    is_uncodable = choice == NO_CODE_OPTION
    is_no_determination = choice == NO_DETERMINATION_OPTION

    if choice == NO_MATCH_OPTION:
        chosen_code = None
        decyzja_kodera_kod = st.text_input(
            "Proszę wpisać poprawny kod ISCO-08",
            key=f"pa_top10_manual_kod_{idx}",
            max_chars=4,
        )
        decyzja_kodera_notatka = st.text_area(
            "Notatka - proszę opisać, o co chodzi w tym przypadku",
            key=f"pa_top10_manual_notatka_{idx}",
            height=80,
        )
        if decyzja_kodera_kod.strip().isdigit() and len(decyzja_kodera_kod.strip()) == 4:
            chosen_code = decyzja_kodera_kod.strip()
    elif is_no_determination:
        chosen_code = fill_code
    elif is_uncodable:
        chosen_code = None
    else:
        choice_idx = options.index(choice)
        chosen_code = ranking.iloc[choice_idx]["isco_code"]

    # Ocena pomocności listy 10 dopasowanych kodów - wymagana zawsze, niezależnie
    # od tego, czy koder wybierze jeden z nich, czy przejdzie do kodowania
    # kaskadowego (patrz przycisk "Kontynuuj kodowanie kaskadowo" niżej).
    ai_helpfulness = render_helpfulness_scale(
        "Jak pomocne były dopasowane kody? (1-5 punktów)",
        key=f"pa_top10_ai_helpfulness_{idx}",
    )

    # Opcjonalna notatka do wyboru - dostępna zawsze, niezależnie od tego, czy
    # koder wybrał jeden z 10 dopasowanych kodów, czy "brak poprawnego kodu"
    # (na wzór pola "Uzasadnienie / komentarz" z trybu kaskadowego).
    uzasadnienie_top10 = st.text_area(
        "Uzasadnienie / komentarz do wyboru (opcjonalnie)",
        key=f"pa_top10_uzasadnienie_{idx}",
        height=70,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Zapisz wybór i przejdź dalej", use_container_width=True, key=f"pa_top10_save_{idx}"):
            if choice == NO_MATCH_OPTION and not chosen_code:
                st.warning("Proszę wpisać poprawny, 4-cyfrowy kod ISCO-08 przed zapisaniem.")
            elif is_uncodable:
                df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5")
                st.session_state[df_state_key] = df
                st.session_state.pop(f"pa1_confirmed_{idx}", None)
                st.session_state.pop(cache_key, None)
                st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, n)
                st.rerun()
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
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5")
                st.session_state[df_state_key] = df
                st.session_state.pop(f"pa1_confirmed_{idx}", None)
                st.session_state.pop(cache_key, None)
                st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, n)
                st.rerun()
    with col_btn2:
        prev_idx = _prev_qualifying_idx(qualifying_positions, idx)
        prev_label = "← Poprzedni partner" if target == "Partner" else "← Poprzedni respondent"
        if prev_idx != idx and st.button(prev_label, use_container_width=True, key=f"pa_top10_prev_{idx}"):
            st.session_state[f"hitl_wracal_{prev_idx}"] = True
            st.session_state.pop(f"pa1_confirmed_{idx}", None)
            st.session_state.pop(cache_key, None)
            st.session_state[idx_state_key] = prev_idx
            st.rerun()

    st.write("")
    if st.button(
        "Kontynuuj kodowanie kaskadowo (cyfry 2-4)",
        key=f"pa_cascade_start_{idx}",
        use_container_width=True,
    ):
        # Zapisujemy ocenę pomocności listy 10 kodów, zanim koder przejdzie
        # do kodowania kaskadowego - inaczej ta ocena nigdy by się nie zapisała.
        df.at[idx, "Ocena_AI_top10_1_5"] = ai_helpfulness
        st.session_state[df_state_key] = df
        st.session_state.pop(f"pa1_confirmed_{idx}", None)
        st.session_state.pop(cache_key, None)
        st.session_state[f"cascade_step_{idx}"] = 2
        st.session_state[f"cascade_digits_{idx}"] = [prefix]
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
        st.session_state.pop("hitl1d_df", None)
        st.session_state.pop("hitl1d_idx", None)
        return

    if "hitl1d_df" not in st.session_state or st.session_state.get("hitl1d_source") != uploaded_file.name:
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
        st.session_state["hitl1d_df"] = df
        st.session_state["hitl1d_idx"] = resume_idx
        st.session_state["hitl1d_source"] = uploaded_file.name
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

    mode_suffix = "respondent" if mode_1digit == "Respondent" else "partner"

    # Pobranie CZĘŚCIOWEGO wyniku - dostępne cały czas w trakcie kodowania
    # (nie trzeba czekać na ukończenie całego pliku). Format pliku jest
    # identyczny jak wynik końcowy, więc częściowy plik można bez przeszkód
    # wgrać z powrotem później - aplikacja sama wznowi kodowanie od pierwszej
    # nieukończonej osoby (patrz _first_unfinished_idx).
    if idx < n:
        with st.expander(f"Pobierz częściowy wynik (dotychczasowy postęp: {progress_rank - 1} z {progress_total})"):
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
        return

    if pa1_confirmed:
        render_pa_top10_step(df, idx, row, prefix=pa1_confirmed, df_state_key="hitl1d_df", idx_state_key="hitl1d_idx")
        return

    render_pa_digit1_step(df, idx, row, df_state_key="hitl1d_df", idx_state_key="hitl1d_idx")


# ============================================================
# TRYB KASKADOWY (kodowanie ISCO-08 od zera, cyfra po cyfrze)
# ============================================================
LEVEL_LABELS = {
    1: "Krok 1 z 4 - grupa główna (1. cyfra)",
    2: "Krok 2 z 4 - grupa drugorzędna (2. cyfra)",
    3: "Krok 3 z 4 - grupa średnia (3. cyfra)",
    4: "Krok 4 z 4 - grupa elementarna (4. cyfra, kod finalny)",
}


def _start_cascade(idx: int):
    st.session_state[f"cascade_step_{idx}"] = 1
    st.session_state[f"cascade_digits_{idx}"] = []


def _cancel_cascade(idx: int):
    st.session_state.pop(f"cascade_step_{idx}", None)
    st.session_state.pop(f"cascade_digits_{idx}", None)


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


def _save_respondent_meta(df, idx: int, ai_helpfulness: Optional[int] = None, ai_column: Optional[str] = None):
    """Zapisuje czas kodowania (sekundy), czas do pierwszej interakcji z listą
    kandydatów, czy koder wracał (cofał się) i ocenę przydatności AI (1-5) -
    w kolumnie zależnej od trybu, którym koder faktycznie kodował tego
    respondenta (ai_column: "Ocena_AI_top10_1_5" albo "Ocena_AI_kaskadowo_1_5")."""
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
        uzasadnienie = ""
        ai_helpfulness = None
        is_no_determination = False
        is_uncodable = False
        chosen_code = None
    else:
        choice = st.radio(
            "Wybierz kod pasujący do tego kroku",
            options=options,
            key=f"cascade_choice_{idx}_{level}_{prefix}",
            on_change=_mark_first_interaction,
            args=(idx,),
        )
        is_no_determination = choice == NO_DETERMINATION_OPTION
        is_uncodable = show_no_code and choice == NO_CODE_OPTION

        if is_no_determination or is_uncodable:
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

        if selected_vars:
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
        else:
            st.caption("Brak zaznaczonych zmiennych w tabeli powyżej (kliknij nazwy kolumn, żeby je zaznaczyć).")

        # Komentarz i ocena AI pojawiają się tylko na ostatnim FAKTYCZNIE
        # osiągniętym poziomie szczegółowości: albo poziom 4, albo moment
        # wyboru "brak możliwości ustalenia dokładnej cyfry" (koniec kodowania).
        show_uzasadnienie = level == 4 or is_no_determination or is_uncodable
        if show_uzasadnienie:
            ai_helpfulness = st.slider(
                "Jak pomocne były dopasowane kody podczas kodowania hierarchicznego?",
                min_value=1,
                max_value=5,
                value=3,
                key=f"cascade_ai_helpfulness_{idx}_{level}_{prefix}",
            )
            uzasadnienie = st.text_area(
                "Uzasadnienie / komentarz do finalnej decyzji (opcjonalnie)",
                key=f"cascade_uzasadnienie_{idx}_{level}_{prefix}",
                height=70,
            )
        else:
            uzasadnienie = ""
            ai_helpfulness = None

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
            df.at[idx, f"ISCO_poziom{level}_zmienne"] = ", ".join(selected_vars) if selected_vars else None
            rank, score = _get_rank_and_score(candidates, chosen_code)
            df.at[idx, f"ISCO_poziom{level}_ranking_pozycja"] = rank
            df.at[idx, f"ISCO_poziom{level}_score"] = score
            if show_uzasadnienie and uzasadnienie:
                df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie.strip()

            # UWAGA: zaznaczenie kolumn w tabeli "Dane respondenta" resetuje się
            # samo przy przejściu na kolejny poziom kaskady, bo klucz widgetu
            # (_resp_table_key) zależy od aktualnego cascade_step_{idx} - nowy
            # poziom = zupełnie nowy widget, bez wcześniejszego zaznaczenia.

            if is_uncodable:
                # Koder jednoznacznie stwierdził, że nie da się zakodować tej
                # osoby do żadnego kodu ISCO-08 - nie wypełniamy cyfr zerami
                # (to celowo inne od "brak możliwości ustalenia" na dalszych
                # krokach), tylko oznaczamy przypadek i przechodzimy dalej.
                df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5")
                st.session_state[df_state_key] = df
                _cancel_cascade(idx)
                st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, n)
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
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5")
                st.session_state[df_state_key] = df
                _cancel_cascade(idx)
                st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, n)
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
                    _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_kaskadowo_1_5")
                    st.session_state[df_state_key] = df
                    _cancel_cascade(idx)
                    st.session_state[idx_state_key] = _next_qualifying_idx(qualifying_positions, idx, n)
                    st.rerun()
                else:
                    st.session_state[df_state_key] = df
                    st.session_state[f"cascade_step_{idx}"] = level + 1
                    st.rerun()


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


def _resp_table_key(idx: int, df_state_key: str) -> str:
    """Klucz widgetu tabeli 'Dane respondenta' (st.dataframe z zaznaczaniem kolumn),
    zależny od aktualnego kroku kaskady (cascade_step_{idx}).

    Dzięki temu przy przejściu do kolejnego lub poprzedniego kroku kaskady tabela
    renderuje się jako CAŁKOWICIE NOWY widget - bez tego, kolumna kliknięta
    (zaznaczona) na jednym kroku zostawałaby "wciśnięta" (podświetlona) również
    na kolejnym kroku, bo Streamlit potrafi zachować stan zaznaczenia widgetu
    po stronie przeglądarki nawet po wyczyszczeniu session_state, dopóki klucz
    widgetu się nie zmienia."""
    level = st.session_state.get(f"cascade_step_{idx}", 0)
    # Namespace modułu zapobiega współdzieleniu stanu tabeli pomiędzy trybem
    # 2 (hitl1d_df) i 3 (hitl_df) dla tego samego numeru respondenta/kroku.
    return f"resp_table_{df_state_key}_{idx}_lvl{level}"


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
        st.session_state.pop("hitl_df", None)
        st.session_state.pop("hitl_idx", None)
        return

    # Wczytanie pliku tylko raz (przy zmianie pliku resetujemy stan)
    if "hitl_df" not in st.session_state or st.session_state.get("hitl_source") != uploaded_file.name:
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
        st.session_state["hitl_df"] = df
        st.session_state["hitl_idx"] = resume_idx
        st.session_state["hitl_source"] = uploaded_file.name
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

    NO_MATCH_OPTION = "Brak poprawnego kodu (decyzja kodera)"
    NO_CODE_OPTION = "Brak możliwości zakodowania do kodu ISCO-08 (przejście do następnej osoby)"
    options = [_format_candidate_label(r.isco_code, r.title_pl, getattr(r, "title_en", ""), r.score) for r in ranking.itertuples()]
    options.append(NO_MATCH_OPTION)
    options.append(NO_CODE_OPTION)

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        key=f"hitl_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None
    decyzja_kodera_kod = ""
    is_uncodable = choice == NO_CODE_OPTION

    if choice == NO_MATCH_OPTION:
        chosen_code = None
        chosen_title = None
        decyzja_kodera_kod = st.text_input(
            "Proszę wpisać poprawny kod ISCO-08",
            key=f"hitl_manual_kod_{idx}",
            max_chars=4,
        )
        decyzja_kodera_notatka = st.text_area(
            "Notatka - proszę opisać, o co chodzi w tym przypadku",
            key=f"hitl_manual_notatka_{idx}",
            height=80,
        )
        if decyzja_kodera_kod.strip().isdigit() and len(decyzja_kodera_kod.strip()) == 4:
            chosen_code = decyzja_kodera_kod.strip()
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
    uzasadnienie_top10 = st.text_area(
        "Uzasadnienie / komentarz do wyboru (opcjonalnie)",
        key=f"hitl_uzasadnienie_{idx}",
        height=70,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Zapisz wybór i przejdź dalej", use_container_width=True):
            if choice == NO_MATCH_OPTION and not chosen_code:
                st.warning("Proszę wpisać poprawny, 4-cyfrowy kod ISCO-08 przed zapisaniem.")
            elif is_uncodable:
                df.at[idx, "Brak_mozliwosci_zakodowania"] = "Tak"
                if uzasadnienie_top10.strip():
                    df.at[idx, "Uzasadnienie_finalne"] = uzasadnienie_top10.strip()
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5")
                st.session_state["hitl_df"] = df
                st.session_state["hitl_idx"] = _next_qualifying_idx(qualifying_positions, idx, n)
                st.rerun()
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
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5")
                st.session_state["hitl_df"] = df
                st.session_state["hitl_idx"] = _next_qualifying_idx(qualifying_positions, idx, n)
                st.rerun()
    with col_btn2:
        prev_idx = _prev_qualifying_idx(qualifying_positions, idx)
        prev_label = "← Poprzedni partner" if target == "Partner" else "← Poprzedni respondent"
        if prev_idx != idx and st.button(prev_label, use_container_width=True):
            st.session_state[f"hitl_wracal_{prev_idx}"] = True
            st.session_state["hitl_idx"] = prev_idx
            st.rerun()

    st.write("")
    if st.button(
        "Zakoduj zawód od zera",
        key=f"cascade_start_{idx}",
        use_container_width=True,
    ):
        # Zapisujemy ocenę pomocności listy 10 kodów, zanim koder przejdzie
        # do kodowania kaskadowego - inaczej ta ocena nigdy by się nie zapisała.
        df.at[idx, "Ocena_AI_top10_1_5"] = ai_helpfulness
        st.session_state["hitl_df"] = df
        _start_cascade(idx)
        st.rerun()


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
    if st.button("Wyloguj", use_container_width=True):
        for key in ("authenticated", "username"):
            st.session_state.pop(key, None)
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
