"""
Eksport korpusu ISCO-08 do struktury folderowej (txt + meta.json) dla WSZYSTKICH
4 poziomów hierarchii, na bazie 4 osobnych plików Excel:

    poziom 1 (grupy główne,        10 kodów) -> Level_1.xlsx
    poziom 2 (grupy drugorzędne,   43 kody)  -> Level_2.xlsx
    poziom 3 (grupy średnie,      130 kodów) -> Level_3.xlsx
    poziom 4 (grupy elementarne,  436 kodów) -> Level_4.xlsx

UWAGA - nowy format plików źródłowych (od wersji z 07.2026):
Każdy plik ma TYLKO 3 pola treściowe (bez Included/Excluded/Notes):
    Nazwa, Kod, Level, Synteza, Zadania zawodowe
Nazwy kolumn różnią się wielkością liter między plikami (np. Level_1.xlsx ma
"nazwa"/"kod"/"level" małymi literami, pozostałe - "Nazwa"/"Kod"/"Level"),
dlatego dopasowanie kolumn jest niewrażliwe na wielkość liter (patrz `_find_col`).

Wynik: 4 osobne foldery korpusu, gotowe do podania na wejście build_embeddings.py:
    isco_corpus_L1/<kod>/...
    isco_corpus_L2/<kod>/...
    isco_corpus_L3/<kod>/...
    isco_corpus_L4/<kod>/...

Odpalić RAZ (lub po każdej zmianie danych źródłowych):

    conda activate isco_rag
    cd ESS_ISCO_APP
    python export_isco_corpus.py
"""

import json
import re
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent

# ============================================================
# KONFIGURACJA POZIOMÓW
# ============================================================
LEVEL_CONFIG = {
    1: {"xlsx": APP_DIR / "Level_1.xlsx", "out_dir": APP_DIR / "isco_corpus_L1"},
    2: {"xlsx": APP_DIR / "Level_2.xlsx", "out_dir": APP_DIR / "isco_corpus_L2"},
    3: {"xlsx": APP_DIR / "Level_3.xlsx", "out_dir": APP_DIR / "isco_corpus_L3"},
    4: {"xlsx": APP_DIR / "Level_4.xlsx", "out_dir": APP_DIR / "isco_corpus_L4"},
}

# Nowy format plików źródłowych - tylko 3 pola treściowe. Klucz = nazwa pliku
# wyjściowego w korpusie, wartość = lista dopuszczalnych nazw kolumn w Excelu
# (dopasowanie niewrażliwe na wielkość liter, bo Level_1.xlsx różni się
# konwencją nazewnictwa kolumn od Level_2/3/4.xlsx).
FIELDS = {
    "title.txt": ["Nazwa"],
    "synteza.txt": ["Synteza"],
    "tasks.txt": ["Zadania zawodowe"],
}

CODE_COL_CANDIDATES = ["Kod"]
LEVEL_COL_CANDIDATES = ["Level"]


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Znajduje nazwę kolumny w df odpowiadającą jednej z `candidates`,
    dopasowując niewrażliwie na wielkość liter (Level_1.xlsx ma kolumny
    zapisane małymi literami, pozostałe pliki - z wielkiej litery)."""
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_and_clean(xlsx_path: str, level: int) -> pd.DataFrame:
    """Wczytuje plik Excel danego poziomu, filtruje (jeśli trzeba) i normalizuje kody."""
    df = pd.read_excel(xlsx_path)

    code_col = _find_col(df, CODE_COL_CANDIDATES)
    level_col = _find_col(df, LEVEL_COL_CANDIDATES)
    assert code_col, f"Nie znaleziono kolumny z kodem ISCO w {xlsx_path} (kolumny: {list(df.columns)})"

    # Jeśli plik zawiera kolumnę "Level" i więcej niż jeden poziom - filtrujemy.
    # Jeśli plik jest już czystym eksportem tylko danego poziomu - zostawiamy jak jest.
    if level_col and df[level_col].nunique() > 1:
        before = len(df)
        df = df.loc[df[level_col] == level].copy()
        print(f"  [poziom {level}] odfiltrowano {before - len(df)} wierszy spoza poziomu {level}")
    else:
        df = df.copy()

    # Normalizacja kodu do dokładnie `level` cyfr (kod bywa liczbą całkowitą, np. 1, 11, 111)
    df["_isco_code"] = (
        df[code_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(level)
    )

    pattern = re.compile(rf"\d{{{level}}}")
    bad_codes = df.loc[~df["_isco_code"].str.fullmatch(pattern), "_isco_code"]
    if len(bad_codes) > 0:
        print(f"  UWAGA [poziom {level}] - kody niepoprawne po zfill({level}):")
        print(" ", bad_codes.tolist())
    else:
        print(f"  OK [poziom {level}] - wszystkie {len(df)} kody mają poprawny format {level}-cyfrowy")

    dupes = df["_isco_code"].duplicated().sum()
    print(f"  [poziom {level}] duplikaty kodów po zfill: {dupes}")

    return df


def export_corpus(data: pd.DataFrame, level: int, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    title_col = _find_col(data, FIELDS["title.txt"])
    synteza_col = _find_col(data, FIELDS["synteza.txt"])
    tasks_col = _find_col(data, FIELDS["tasks.txt"])

    for _, row in data.iterrows():
        code = row["_isco_code"]
        assert len(code) == level and code.isdigit(), f"Niepoprawny kod ISCO (poziom {level}): {code!r}"

        code_dir = out_path / code
        code_dir.mkdir(exist_ok=True)

        (code_dir / "title.txt").write_text(safe_text(row[title_col]) if title_col else "", encoding="utf-8")
        (code_dir / "synteza.txt").write_text(safe_text(row[synteza_col]) if synteza_col else "", encoding="utf-8")
        (code_dir / "tasks.txt").write_text(safe_text(row[tasks_col]) if tasks_col else "", encoding="utf-8")

        meta = {
            "isco_code": code,
            "level": level,
            "title_pl": safe_text(row[title_col]) if title_col else "",
            # skill_level = pierwsza cyfra kodu (zgodnie z dotychczasową konwencją)
            "skill_level": int(code[0]),
        }
        (code_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"  Wyeksportowano {len(data)} folderów do: {out_path.resolve()}")


def main():
    for level, cfg in LEVEL_CONFIG.items():
        print(f"\n=== Poziom {level}: {cfg['xlsx']} ===")
        if not Path(cfg["xlsx"]).exists():
            print(f"  POMINIĘTO - brak pliku {cfg['xlsx']} w folderze roboczym")
            continue
        df = load_and_clean(cfg["xlsx"], level)
        export_corpus(df, level, cfg["out_dir"])

    print("\nGotowe (pola: title/synteza/tasks). Teraz odpal build_embeddings.py, żeby zwektoryzować wszystkie 4 poziomy.")


if __name__ == "__main__":
    main()
