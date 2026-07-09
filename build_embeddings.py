"""
Budowa baz embeddingów ISCO-08 (title, tasks, synteza) zapisanych jako pliki .npy,
DLA WSZYSTKICH 4 POZIOMÓW hierarchii (10 / 43 / 130 / 436 kodów).

UWAGA: od wersji z 07.2026 pole "included" zniknęło ze źródłowych Excelów
(Level_1..4.xlsx zawierają teraz tylko Nazwa/Synteza/Zadania zawodowe),
dlatego trzeci embedding to teraz "synteza" (odpowiednik dawnego Definition PL).

Embeddingi liczone są modelem Qwen3-Embedding-8B, uruchomionym LOKALNIE przez Ollama
(brak API, brak wysyłania danych na zewnątrz - zgodnie z RODO). Model jest obecnie
najlepszym dostępnym lokalnie modelem embeddingowym dla wyszukiwania semantycznego
(#1 na multilingual MTEB leaderboard, dobrze radzi sobie też z polskim).

WYMAGANIA PRZED URUCHOMIENIEM:
    1. Ollama musi być zainstalowana i uruchomiona lokalnie (masz ją już do Bielika).
    2. Pobierz model (jednorazowo, ~5-6 GB):
           ollama pull qwen3-embedding:8b
    3. Doinstaluj klienta Pythona do Ollama w swoim środowisku:
           conda activate isco_rag
           pip install ollama

Wymaga wcześniejszego uruchomienia export_isco_corpus.py (tworzy isco_corpus_L1..L4).

Jeśli w folderze roboczym jest plik Opisy.xlsx (oryginalna struktura ISCO-08 EN,
arkusz "ISCO-08 EN Struct and defin"), do metadata.json każdego poziomu zostanie
dopisane pole "title_en" (angielski odpowiednik nazwy zawodu) - wykorzystywane
w app.py do wyświetlania obok polskiej nazwy. Jeśli już masz zbudowane embeddingi
i chcesz dołożyć tylko title_en bez ponownego (kosztownego) liczenia embeddingów,
użyj zamiast tego skryptu add_english_titles.py.

Odpalić RAZ (lub po każdej zmianie danych w isco_corpus_L*/):

    conda activate isco_rag
    cd ESS_ISCO_APP
    python build_embeddings.py
"""

import json
from pathlib import Path

import numpy as np
import ollama
import pandas as pd

APP_DIR = Path(__file__).resolve().parent

OLLAMA_MODEL = "qwen3-embedding:8b"
BATCH_SIZE = 32  # ile tekstów wysyłamy do Ollama w jednym zapytaniu /api/embed

# Oryginalny plik struktury ISCO-08 (angielskie nazwy zawodów) - opcjonalny:
# jeśli jest obecny w folderze roboczym, do metadata.json każdego poziomu
# dopisywane jest pole "title_en" (patrz load_english_titles / build_level).
OPISY_XLSX = APP_DIR / "Opisy.xlsx"
OPISY_SHEET = "ISCO-08 EN Struct and defin"

# poziom -> (folder korpusu wejściowego, folder embeddingów wyjściowego)
LEVEL_CONFIG = {
    1: {"corpus_dir": APP_DIR / "isco_corpus_L1", "out_dir": APP_DIR / "isco_embeddings" / "level_1"},
    2: {"corpus_dir": APP_DIR / "isco_corpus_L2", "out_dir": APP_DIR / "isco_embeddings" / "level_2"},
    3: {"corpus_dir": APP_DIR / "isco_corpus_L3", "out_dir": APP_DIR / "isco_embeddings" / "level_3"},
    4: {"corpus_dir": APP_DIR / "isco_corpus_L4", "out_dir": APP_DIR / "isco_embeddings" / "level_4"},
}


class OllamaEmbedder:
    """Cienki wrapper na lokalne API Ollama, naśladujący interfejs
    SentenceTransformer.encode() używany w reszcie kodu (app.py)."""

    def __init__(self, model_name: str = OLLAMA_MODEL, batch_size: int = BATCH_SIZE):
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
        n = len(texts)
        for start in range(0, n, self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = ollama.embed(model=self.model_name, input=batch)
            all_embeddings.extend(response.embeddings)
            if show_progress_bar:
                done = min(start + self.batch_size, n)
                print(f"    ... {done}/{n}")

        embeddings = np.array(all_embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings


def load_corpus(corpus_dir: str) -> dict:
    corpus = {}
    for code_dir in sorted(Path(corpus_dir).iterdir()):
        if not code_dir.is_dir():
            continue
        meta = json.loads((code_dir / "meta.json").read_text(encoding="utf-8"))
        corpus[meta["isco_code"]] = {
            "title": (code_dir / "title.txt").read_text(encoding="utf-8"),
            "tasks": (code_dir / "tasks.txt").read_text(encoding="utf-8"),
            "synteza": (code_dir / "synteza.txt").read_text(encoding="utf-8"),
            "skill_level": meta["skill_level"],
        }
    return corpus


def load_english_titles() -> dict:
    """Zwraca słownik {kod_isco (string, wyzerowany do właściwej długości): Title EN}
    dla wszystkich 4 poziomów naraz, na bazie oryginalnego pliku struktury ISCO-08
    (Opisy.xlsx, kolumny Level / ISCO 08 Code / Title EN). Zwraca pusty słownik,
    jeśli plik nie istnieje w folderze roboczym - wtedy metadata.json po prostu
    nie dostanie pola title_en (patrz build_level)."""
    if not Path(OPISY_XLSX).exists():
        return {}
    df = pd.read_excel(OPISY_XLSX, sheet_name=OPISY_SHEET)
    titles_en = {}
    for _, row in df.iterrows():
        level = int(row["Level"])
        code = str(int(row["ISCO 08 Code"])).zfill(level)
        titles_en[code] = str(row["Title EN"]).strip() if pd.notna(row["Title EN"]) else ""
    return titles_en


def build_level(level: int, corpus_dir: str, out_dir: str, model: OllamaEmbedder, titles_en: dict):
    print(f"\n=== Poziom {level}: {corpus_dir} ===")
    if not Path(corpus_dir).exists():
        print(f"  POMINIĘTO - brak folderu {corpus_dir} (odpal najpierw export_isco_corpus.py)")
        return

    corpus = load_corpus(corpus_dir)
    codes_ordered = list(corpus.keys())
    print(f"  Wczytano {len(codes_ordered)} kodów ISCO (poziom {level})")

    def embed_field(field: str) -> np.ndarray:
        texts = [corpus[code][field] for code in codes_ordered]
        # UWAGA: teksty korpusu ISCO (title/tasks/synteza) NIE dostają prefiksu
        # instrukcji - to są "dokumenty", nie "zapytania". Prefiks dokładamy
        # tylko po stronie zapytania w app.py (patrz QUERY_INSTRUCTION).
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

    print("  Generowanie embeddingów: title...")
    title_emb = embed_field("title")
    print("  Generowanie embeddingów: tasks...")
    tasks_emb = embed_field("tasks")
    print("  Generowanie embeddingów: synteza...")
    synteza_emb = embed_field("synteza")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "title_emb.npy", title_emb)
    np.save(out_path / "tasks_emb.npy", tasks_emb)
    np.save(out_path / "synteza_emb.npy", synteza_emb)

    with open(out_path / "codes_ordered.json", "w", encoding="utf-8") as f:
        json.dump(codes_ordered, f, ensure_ascii=False, indent=2)

    metadata = {
        code: {
            "title": corpus[code]["title"],
            "title_en": titles_en.get(code, ""),
            "skill_level": corpus[code]["skill_level"],
        }
        for code in codes_ordered
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    matched_en = sum(1 for code in codes_ordered if titles_en.get(code))
    print(f"  Zapisano embeddingi i metadane w folderze: {out_path.resolve()}")
    if titles_en:
        print(f"  Angielskie nazwy (title_en): dopasowano {matched_en}/{len(codes_ordered)} kodów")
    else:
        print(f"  UWAGA: brak pliku {OPISY_XLSX} - metadata.json bez pola title_en (dopasuj potem add_english_titles.py)")


def main():
    print(f"Model embeddingowy: {OLLAMA_MODEL} (lokalnie przez Ollama)")
    print("Upewnij się, że Ollama jest uruchomiona i masz pobrany model:")
    print(f"  ollama pull {OLLAMA_MODEL}\n")

    model = OllamaEmbedder()
    titles_en = load_english_titles()
    if titles_en:
        print(f"Wczytano {len(titles_en)} angielskich nazw zawodów z {OPISY_XLSX}\n")
    else:
        print(f"UWAGA: nie znaleziono pliku {OPISY_XLSX} - metadata.json nie będzie mieć pola title_en.\n")

    for level, cfg in LEVEL_CONFIG.items():
        build_level(level, cfg["corpus_dir"], cfg["out_dir"], model, titles_en)

    print("\nGotowe - wszystkie poziomy zwektoryzowane.")


if __name__ == "__main__":
    main()
