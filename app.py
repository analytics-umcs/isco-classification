from __future__ import annotations

import io
import json
import hmac
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import ollama

# Ścieżka do folderu roboczego projektu - ustawiona na sztywno, żeby ścieżki
# względne (isco_embeddings/...) działały niezależnie od tego, skąd odpalasz
# "streamlit run app.py". Zmień, jeśli projekt leży gdzie indziej.
WORKDIR = "/Users/mblaszczykowski/Desktop/App_2Mod_RP2"
if os.path.isdir(WORKDIR):
    os.chdir(WORKDIR)

# ============================================================
# KONFIGURACJA STRONY
# ============================================================
st.set_page_config(
    page_title="Klasyfikacja zawodów ISCO-08",
    page_icon="🧭",
    layout="centered",
)

EMB_DIR = "isco_embeddings/level_4"  # pełna baza 436 kodów (poziom 4) - używana w modułach 1 i 2
MODEL_NAME = "qwen3-embedding:8b"  # lokalnie przez Ollama, brak API
OLLAMA_BATCH_SIZE = 32

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
    1: "isco_embeddings/level_1",   # 10 kodów - grupy główne
    2: "isco_embeddings/level_2",   # 43 kody - grupy drugorzędne
    3: "isco_embeddings/level_3",   # 130 kodów - grupy średnie
    4: "isco_embeddings/level_4",   # 436 kodów - grupy elementarne
}

# Kolumny źródłowe używane do klasyfikacji - osobny zestaw dla respondenta
# i dla jego partnera. Przełącznik "Respondent / Partner" (patrz
# render_mode_selector) decyduje, z którego zestawu korzystają moduły 1 i 2.
TARGET_COLUMNS = {
    "Respondent": {"zawod": "B33", "obowiazki": "B34", "wyksztalcenie": "B35"},
    "Partner": {"zawod": "B48", "obowiazki": "B49", "wyksztalcenie": "B50"},
}


# Kolory identyfikacji wizualnej
COLOR_ISCO = "#003B73"  # granat
COLOR_ESS = "#C1121F"   # czerwony

CUSTOM_CSS = f"""
<style>
.top-bar {{
    height: 6px;
    width: 100%;
    background: linear-gradient(to right, {COLOR_ISCO} 0%, {COLOR_ISCO} 50%, {COLOR_ESS} 50%, {COLOR_ESS} 100%);
    margin-bottom: 1.5rem;
    border-radius: 3px;
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
</style>
"""


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
    "Respondent": "ess_var_metadata_pl_respondent.json",
    "Partner": "ess_var_metadata_pl_partner.json",
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
    nie istniał) na stałe między rerunami, mimo późniejszej zmiany WORKDIR
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
    return df[keep_cols]


def _build_column_help(col: str, var_meta: dict) -> str | None:
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


def _reset_module_progress(df_state_key: str, idx_state_key: str):
    """Czyści cały postęp kodowania w danym module (wybory, cache klasyfikacji,
    liczniki czasu, stan kaskady, zaznaczenia w tabeli) i cofa do respondenta
    nr 1. Używane przy twardym przełączeniu trybu Respondent/Partner w trakcie
    sesji. Same dane (df) i wynik zapisany w kolumnach wynikowych NIE są
    czyszczone - o ich pobranie (częściowy CSV) prosimy PRZED przełączeniem."""
    clear_prefixes = ("hitl_", "hitl1d_", "cascade_", "pa1_", "pa_top10_", "resp_table_")
    keep_keys = {
        df_state_key,
        idx_state_key,
        "hitl_source", "hitl1d_source",
        "hitl_source_cols", "hitl1d_source_cols",
        _mode_key(df_state_key),
    }
    for key in list(st.session_state.keys()):
        if key in keep_keys:
            continue
        if key.startswith(clear_prefixes):
            del st.session_state[key]
    st.session_state[idx_state_key] = 0


def render_mode_selector(df_state_key: str, idx_state_key: str, df, idx: int, n: int) -> str:
    """Renderuje dwa przyciski 'Koduj respondentów' / 'Koduj partnerów'.
    Wybór dotyczy CAŁEGO wczytanego pliku, nie tylko aktualnie wyświetlanego
    wiersza. Jeśli kodowanie w bieżącym trybie jest już W TRAKCIE (nie
    ukończono jeszcze wszystkich respondentów), kliknięcie drugiego trybu NIE
    przełącza od razu - pokazuje ostrzeżenie z możliwością pobrania
    częściowego wyniku i zakończenia bieżącej sesji. Zakończenie NIE przełącza
    automatycznie na drugi tryb - to osobna, świadoma decyzja kodera."""
    mode_key = _mode_key(df_state_key)
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Respondent"
    current = st.session_state[mode_key]

    in_progress = 0 < idx < n
    pending_key = f"pending_mode_{df_state_key}"

    if st.session_state.get(pending_key):
        current_plural = "respondentów" if current == "Respondent" else "partnerów"

        st.warning(
            f"Kodowanie {current_plural} nie zostało jeszcze ukończone ({idx} z {n}). "
            "Żeby przełączyć się na drugi tryb, najpierw zakończ i zapisz obecną sesję - "
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
                st.session_state.pop(pending_key, None)
                _reset_module_progress(df_state_key, idx_state_key)
                st.rerun()
        with col_cancel:
            if st.button(
                "Anuluj, wróć do kodowania",
                use_container_width=True,
                key=f"cancel_switch_{df_state_key}",
            ):
                st.session_state.pop(pending_key, None)
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
                else:
                    st.session_state[mode_key] = "Respondent"
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
                else:
                    st.session_state[mode_key] = "Partner"
                st.rerun()

    return st.session_state[mode_key]





def _decode_value(raw_val, value_labels: dict) -> str | None:
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
    prefix: str | None = None,
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
    prefix: str | None = None,
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
                "score": round(float(row_scores[i]), 4),
            }
            for i in top_idx
        ]
        results.append(candidates)

    return results


def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """Wczytuje CSV z autodetekcją separatora i kodowania."""
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None


def go_to(page_name: str):
    st.session_state.page = page_name


def _valid_login(username: str, password: str) -> bool:
    expected_password = APP_USERS.get(username)
    if expected_password is None:
        return False
    return hmac.compare_digest(password, expected_password)


def render_login():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-header">
            <h1>System wspomagania klasyfikacji zawodów ISCO-08</h1>
            <p>Logowanie</p>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Użytkownik")
        password = st.text_input("Hasło", type="password")
        submitted = st.form_submit_button("Zaloguj", use_container_width=True)

    if submitted:
        username = username.strip()
        if _valid_login(username, password):
            st.session_state.authenticated = True
            st.session_state.authenticated_user = username
            st.session_state.page = "home"
            st.rerun()
        else:
            st.error("Nieprawidłowy użytkownik lub hasło.")


def require_login():
    if not st.session_state.authenticated:
        render_login()
        st.stop()

    st.sidebar.caption(f"Zalogowano jako: {st.session_state.authenticated_user}")
    if st.sidebar.button("Wyloguj", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.session_state.page = "home"
        st.rerun()


# ============================================================
# STRONA GŁÓWNA
# ============================================================
def render_home():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
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

    st.write(
        "Aplikacja umożliwia klasyfikację zawodów zgodnie ze standardem ISCO-08 "
        "z wykorzystaniem modeli sztucznej inteligencji, na podstawie danych ankietowych "
        "European Social Survey (ESS). System wspiera klasyfikację zawodów wspomaganą "
        "decyzją eksperta."
    )

    st.write("")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta</div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_hitl", use_container_width=True):
                go_to("classify_hitl")

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="module-card-title">Klasyfikacja zawodów<br>z udziałem&nbsp;eksperta'
                '<span class="module-card-sub">(1&nbsp;cyfra&nbsp;przyporządkowana)</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Otwórz", key="btn_hitl_1digit", use_container_width=True):
                go_to("classify_hitl_1digit")


# ============================================================
# STRONA: KLASYFIKACJA Z UDZIAŁEM EKSPERTA (1 CYFRA PRZYPORZĄDKOWANA)
# ============================================================
# Kolumna z danych wejściowych, w której przechowywany jest kod ISCO-08
# wcześniej przyporządkowany respondentowi (np. przez system automatyczny),
# ale TYLKO dla 1. cyfry - pozostałe cyfry (2, 3, 4) nie są przyporządkowane
# i są dokodowywane tak samo jak w pełnej kaskadzie (AI proponuje kandydatów).
PA_DIGIT_COLUMNS = {1: "ISCO_1Digit_Respondent"}


def _normalize_isco_digit_code(raw_value, level: int) -> str | None:
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
    title_txt = f" — {_lower_first(proposed_title)}" if proposed_title else ""
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
    options = [f"{r.isco_code} — {_lower_first(r.title_pl)} (dopasowanie: {r.score:.3f})" for r in ranking.itertuples()]
    options.append(NO_MATCH_OPTION)

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        key=f"pa_top10_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None

    if choice == NO_MATCH_OPTION:
        chosen_code = None
        decyzja_kodera_zawod = st.text_input(
            "Proszę wpisać poprawny zawód", key=f"pa_top10_manual_zawod_{idx}"
        )
        decyzja_kodera_notatka = st.text_area(
            "Notatka - proszę opisać, o co chodzi w tym przypadku",
            key=f"pa_top10_manual_notatka_{idx}",
            height=80,
        )
    else:
        choice_idx = options.index(choice)
        chosen_code = ranking.iloc[choice_idx]["isco_code"]

    # Ocena pomocności listy 10 dopasowanych kodów - wymagana zawsze, niezależnie
    # od tego, czy koder wybierze jeden z nich, czy przejdzie do kodowania
    # kaskadowego (patrz przycisk "Kontynuuj kodowanie kaskadowo" niżej).
    ai_helpfulness = st.slider(
        "Jak pomocne było 10 dopasowanych kodów?",
        min_value=1,
        max_value=5,
        value=3,
        key=f"pa_top10_ai_helpfulness_{idx}",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Zapisz wybór i przejdź dalej", use_container_width=True, key=f"pa_top10_save_{idx}"):
            if choice == NO_MATCH_OPTION and not decyzja_kodera_zawod.strip():
                st.warning("Proszę wpisać poprawny zawód przed zapisaniem.")
            else:
                df.at[idx, "ISCO_wybrany"] = chosen_code
                df.at[idx, "Decyzja_kodera_zawod"] = decyzja_kodera_zawod
                df.at[idx, "Decyzja_kodera_notatka"] = decyzja_kodera_notatka
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
                st.session_state[idx_state_key] = idx + 1
                st.rerun()
    with col_btn2:
        prev_label = "← Poprzedni partner" if target == "Partner" else "← Poprzedni respondent"
        if idx > 0 and st.button(prev_label, use_container_width=True, key=f"pa_top10_prev_{idx}"):
            st.session_state[f"hitl_wracal_{idx - 1}"] = True
            st.session_state.pop(f"pa1_confirmed_{idx}", None)
            st.session_state.pop(cache_key, None)
            st.session_state[idx_state_key] = idx - 1
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

        st.session_state["hitl_source_cols"] = df.select_dtypes(include="number").columns.tolist()

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

        st.session_state["hitl1d_df"] = df
        st.session_state["hitl1d_idx"] = 0
        st.session_state["hitl1d_source"] = uploaded_file.name

    df = st.session_state["hitl1d_df"]
    idx = st.session_state["hitl1d_idx"]
    n = len(df)

    st.write("")
    render_mode_selector("hitl1d_df", "hitl1d_idx", df, idx, n)
    mode_1digit = _get_coding_target("hitl1d_df")
    _warn_if_meta_missing(mode_1digit)
    st.write("")
    podmiot_label = "Partner" if mode_1digit == "Partner" else "Respondent"
    st.progress((idx) / n if n > 0 else 0, text=f"{podmiot_label} {min(idx + 1, n)} z {n}")

    if idx >= n:
        podmiot_plural = "partnerów" if mode_1digit == "Partner" else "respondentów"
        st.success(f"Zweryfikowano wszystkich {podmiot_plural}.")

        mode_suffix = "respondent" if mode_1digit == "Respondent" else "partner"
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="wyniki")
        xlsx_bytes = xlsx_buffer.getvalue()

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

    st.write("Dane respondenta:")
    st.caption("Kliknij nazwy kolumn (zmiennych), z których korzystasz przy klasyfikacji.")

    mode_1d_row = _get_coding_target("hitl1d_df")
    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_1d_row),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, load_var_metadata(mode_1d_row)),
        on_select="rerun",
        selection_mode=["multi-column"],
        key=_resp_table_key(idx),
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


def _get_rank_and_score(candidates: pd.DataFrame, chosen_code) -> tuple[int | None, float | None]:
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


def _save_respondent_meta(df, idx: int, ai_helpfulness: int | None = None, ai_column: str | None = None):
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

    options = [
        f"{r.isco_code} — {_lower_first(r.title_pl)} (dopasowanie: {r.score:.3f})"
        for r in candidates.itertuples()
    ]
    if show_no_determination:
        options = options + [NO_DETERMINATION_OPTION]

    if not options:
        st.warning(
            "Brak kodów ISCO-08 pasujących do wybranego dotychczas prefiksu. "
            "Cofnij się o krok i wybierz inną cyfrę."
        )
        selected_vars = []
        uzasadnienie = ""
        ai_helpfulness = None
        is_no_determination = False
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

        if is_no_determination:
            chosen_code = None
            chosen_title = None
        else:
            choice_idx = options.index(choice)
            chosen_code = candidates.iloc[choice_idx]["isco_code"]
            chosen_title = candidates.iloc[choice_idx]["title_pl"]

        # Zmienne, z których korzystał koder = kolumny zaznaczone kliknięciem
        # w tabeli "Dane respondenta" powyżej (współdzielony stan z tą tabelą),
        # zawężone do zmiennych nietekstowych.
        source_cols = st.session_state.get("hitl_source_cols", [])
        var_meta = load_var_metadata(target)
        table_selection = st.session_state.get(_resp_table_key(idx), {})
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
        show_uzasadnienie = level == 4 or is_no_determination
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
        next_label = "Zatwierdź kod finalny" if (level == 4 or is_no_determination) else "Dalej →"
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

            if is_no_determination:
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
                st.session_state[idx_state_key] = idx + 1
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
                    st.session_state[idx_state_key] = idx + 1
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


def _resp_table_key(idx: int) -> str:
    """Klucz widgetu tabeli 'Dane respondenta' (st.dataframe z zaznaczaniem kolumn),
    zależny od aktualnego kroku kaskady (cascade_step_{idx}).

    Dzięki temu przy przejściu do kolejnego lub poprzedniego kroku kaskady tabela
    renderuje się jako CAŁKOWICIE NOWY widget - bez tego, kolumna kliknięta
    (zaznaczona) na jednym kroku zostawałaby "wciśnięta" (podświetlona) również
    na kolejnym kroku, bo Streamlit potrafi zachować stan zaznaczenia widgetu
    po stronie przeglądarki nawet po wyczyszczeniu session_state, dopóki klucz
    widgetu się nie zmienia."""
    level = st.session_state.get(f"cascade_step_{idx}", 0)
    return f"resp_table_{idx}_lvl{level}"


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

        # Zapamiętujemy oryginalne kolumny z pliku - to z nich koder będzie (opcjonalnie)
        # wybierał, z których zmiennych tabelarycznych korzystał przy każdej decyzji
        # kaskadowej. Tylko zmienne NIEtekstowe (numeryczne) - B33/B34/B35 i B48/B49/B50
        # to wolny tekst używany bezpośrednio do embeddingu, więc nie mają tu sensu.
        st.session_state["hitl_source_cols"] = df.select_dtypes(include="number").columns.tolist()

        if "ISCO_wybrany" not in df.columns:
            df["ISCO_wybrany"] = None
            df["Decyzja_kodera_zawod"] = None
            df["Decyzja_kodera_notatka"] = None
        if "Kodowany_podmiot" not in df.columns:
            df["Kodowany_podmiot"] = None
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

        st.session_state["hitl_df"] = df
        st.session_state["hitl_idx"] = 0
        st.session_state["hitl_source"] = uploaded_file.name

    df = st.session_state["hitl_df"]
    idx = st.session_state["hitl_idx"]
    n = len(df)

    st.write("")
    render_mode_selector("hitl_df", "hitl_idx", df, idx, n)
    mode_main = _get_coding_target("hitl_df")
    _warn_if_meta_missing(mode_main)
    st.write("")
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        podmiot_label = "Partner" if mode_main == "Partner" else "Respondent"
        st.progress((idx) / n if n > 0 else 0, text=f"{podmiot_label} {min(idx + 1, n)} z {n}")
    with col_nav2:
        with st.popover("Podgląd danych"):
            visible_df = visible_df_for_mode(df, mode_main)
            st.dataframe(visible_df, use_container_width=True, column_config=build_column_config(visible_df, load_var_metadata(mode_main)))

    if idx >= n:
        podmiot_plural = "partnerów" if mode_main == "Partner" else "respondentów"
        st.success(f"Sklasyfikowano wszystkich {podmiot_plural}.")

        ref_cols = [c for c in ["B33", "B34", "B35", "B48", "B49", "B50", "ISCO08"] if c in df.columns]
        result_cols = [
            "Kodowany_podmiot",
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
        final_df = df[other_cols + ref_cols + result_cols].copy()

        csv_bytes = final_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="wyniki")
            worksheet = writer.sheets["wyniki"]
            for i, col in enumerate(final_df.columns, start=1):
                max_len = max(
                    final_df[col].apply(lambda v: len(str(v)) if pd.notna(v) else 0).max() if len(final_df) else 0,
                    len(str(col)),
                )
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len + 2, 60)
        xlsx_bytes = xlsx_buffer.getvalue()

        mode_suffix = "respondent" if mode_main == "Respondent" else "partner"

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

    st.write("Dane respondenta:")
    st.caption("Kliknij nazwy kolumn (zmiennych), z których korzystasz przy klasyfikacji.")

    _var_meta_debug = load_var_metadata(mode_main)

    st.dataframe(
        visible_df_for_mode(df.iloc[[idx]], mode_main),
        use_container_width=True,
        column_config=build_column_config_for_respondent(row, _var_meta_debug),
        on_select="rerun",
        selection_mode=["multi-column"],
        key=_resp_table_key(idx),
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
    options = [f"{r.isco_code} — {_lower_first(r.title_pl)} (dopasowanie: {r.score:.3f})" for r in ranking.itertuples()]
    options.append(NO_MATCH_OPTION)

    choice = st.radio(
        "Wybierz właściwy kod ISCO-08",
        options=options,
        key=f"hitl_choice_{idx}",
        on_change=_mark_first_interaction,
        args=(idx,),
    )

    decyzja_kodera_zawod = None
    decyzja_kodera_notatka = None

    if choice == NO_MATCH_OPTION:
        chosen_code = None
        chosen_title = None
        decyzja_kodera_zawod = st.text_input(
            "Proszę wpisać poprawny zawód", key=f"hitl_manual_zawod_{idx}"
        )
        decyzja_kodera_notatka = st.text_area(
            "Notatka - proszę opisać, o co chodzi w tym przypadku",
            key=f"hitl_manual_notatka_{idx}",
            height=80,
        )
    else:
        choice_idx = options.index(choice)
        chosen_code = ranking.iloc[choice_idx]["isco_code"]
        chosen_title = ranking.iloc[choice_idx]["title_pl"]

    # Ocena pomocności listy 10 dopasowanych kodów - wymagana zawsze, niezależnie
    # od tego, czy koder wybierze jeden z nich, czy przejdzie do kodowania
    # kaskadowego (patrz przycisk "Zakoduj od zera" niżej).
    ai_helpfulness = st.slider(
        "Jak pomocne było 10 dopasowanych kodów?",
        min_value=1,
        max_value=5,
        value=3,
        key=f"hitl_ai_helpfulness_{idx}",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Zapisz wybór i przejdź dalej", use_container_width=True):
            if choice == NO_MATCH_OPTION and not decyzja_kodera_zawod.strip():
                st.warning("Proszę wpisać poprawny zawód przed zapisaniem.")
            else:
                df.at[idx, "ISCO_wybrany"] = chosen_code
                df.at[idx, "Decyzja_kodera_zawod"] = decyzja_kodera_zawod
                df.at[idx, "Decyzja_kodera_notatka"] = decyzja_kodera_notatka
                rank, score = _get_rank_and_score(ranking, chosen_code)
                df.at[idx, "Ranking_pozycja_wybranego_kodu"] = rank
                df.at[idx, "Score_wybranego_kodu"] = score
                df.at[idx, "Kodowany_podmiot"] = target
                _save_respondent_meta(df, idx, ai_helpfulness, ai_column="Ocena_AI_top10_1_5")
                st.session_state["hitl_df"] = df
                st.session_state["hitl_idx"] = idx + 1
                st.rerun()
    with col_btn2:
        prev_label = "← Poprzedni partner" if target == "Partner" else "← Poprzedni respondent"
        if idx > 0 and st.button(prev_label, use_container_width=True):
            st.session_state[f"hitl_wracal_{idx - 1}"] = True
            st.session_state["hitl_idx"] = idx - 1
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

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "classify_hitl":
    render_classify_hitl()
elif st.session_state.page == "classify_hitl_1digit":
    render_classify_hitl_1digit()
