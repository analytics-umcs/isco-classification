"""
Dopisuje angielskie nazwy zawodów (Title EN) z oryginalnego pliku struktury
ISCO-08 (Opisy.xlsx, arkusz "ISCO-08 EN Struct and defin") do JUŻ ISTNIEJĄCYCH
plików metadata.json (wszystkie 4 poziomy) - BEZ ponownego liczenia embeddingów.

Kiedy tego użyć: masz już zbudowane isco_embeddings/level_1..4/ (przez
build_embeddings.py) i chcesz tylko dołożyć angielskie odpowiedniki nazw,
które app.py pokazuje obok polskich tytułów w listach wyboru.

Jeśli wolisz, żeby title_en pojawiało się automatycznie przy KAŻDEJ pełnej
przebudowie embeddingów, wystarczy mieć plik Opisy.xlsx w folderze roboczym
przed odpaleniem build_embeddings.py - on też potrafi to zrobić sam. Ten
skrypt jest szybszym rozwiązaniem "na już", bo nie woła w ogóle Ollamy.

WYMAGANIA:
    - plik Opisy.xlsx w folderze roboczym projektu (ten sam, co Level_1..4.xlsx)
    - wcześniej zbudowane isco_embeddings/level_1..4/metadata.json

Odpalić RAZ (albo ponownie, jeśli zmieni się zawartość Opisy.xlsx):

    conda activate isco_rag
    cd ESS_ISCO_APP
    python add_english_titles.py
"""

import json
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent

OPISY_XLSX = APP_DIR / "Opisy.xlsx"
OPISY_SHEET = "ISCO-08 EN Struct and defin"

LEVEL_META_PATHS = {
    1: APP_DIR / "isco_embeddings" / "level_1" / "metadata.json",
    2: APP_DIR / "isco_embeddings" / "level_2" / "metadata.json",
    3: APP_DIR / "isco_embeddings" / "level_3" / "metadata.json",
    4: APP_DIR / "isco_embeddings" / "level_4" / "metadata.json",
}


def load_english_titles() -> dict:
    """Zwraca słownik {kod_isco (string, wyzerowany do właściwej długości): Title EN}
    dla wszystkich 4 poziomów naraz, na bazie kolumn Level / ISCO 08 Code / Title EN."""
    df = pd.read_excel(OPISY_XLSX, sheet_name=OPISY_SHEET)
    titles_en = {}
    for _, row in df.iterrows():
        level = int(row["Level"])
        code = str(int(row["ISCO 08 Code"])).zfill(level)
        titles_en[code] = str(row["Title EN"]).strip() if pd.notna(row["Title EN"]) else ""
    return titles_en


def main():
    if not Path(OPISY_XLSX).exists():
        print(f"BŁĄD: nie znaleziono pliku {OPISY_XLSX} w folderze roboczym.")
        print(f"Bieżący katalog roboczy: {Path.cwd()}")
        xlsx_here = [p.name for p in Path.cwd().glob("*.xlsx")]
        print(f"Pliki .xlsx w tym katalogu: {xlsx_here if xlsx_here else 'brak'}")
        return

    titles_en = load_english_titles()
    print(f"Wczytano {len(titles_en)} angielskich nazw (wszystkie poziomy) z {OPISY_XLSX}")

    for level, meta_path in LEVEL_META_PATHS.items():
        path = Path(meta_path)
        if not path.exists():
            print(f"  POMINIĘTO poziom {level} - brak pliku {meta_path} (odpal najpierw build_embeddings.py)")
            continue

        metadata = json.loads(path.read_text(encoding="utf-8"))
        matched = 0
        for code, entry in metadata.items():
            title_en = titles_en.get(code)
            if title_en:
                entry["title_en"] = title_en
                matched += 1
            else:
                entry.setdefault("title_en", "")

        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Poziom {level}: dopasowano {matched}/{len(metadata)} kodów -> zapisano {meta_path}")

    print("\nGotowe - metadata.json wszystkich poziomów mają teraz pole 'title_en'.")
    print("Odśwież (lub uruchom ponownie) aplikację Streamlit, żeby zobaczyć zmiany.")


if __name__ == "__main__":
    main()
