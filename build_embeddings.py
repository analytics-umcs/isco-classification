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

Odpalić RAZ (lub po każdej zmianie danych w isco_corpus_L*/):

    conda activate isco_rag
    cd ~/Desktop/App_2Mod_RP2
    python build_embeddings.py
"""

import json
import os
from pathlib import Path

import numpy as np
import ollama

# Ścieżka do folderu roboczego projektu - ustawiona na sztywno, żeby nie trzeba
# było pamiętać o "cd" przed odpaleniem skryptu. Zmień, jeśli projekt leży gdzie indziej.
WORKDIR = "/Users/mblaszczykowski/Desktop/App_2Mod_RP2"
if os.path.isdir(WORKDIR):
    os.chdir(WORKDIR)
else:
    print(f"UWAGA: nie znaleziono folderu {WORKDIR} - zostaję w bieżącym katalogu roboczym.")

OLLAMA_MODEL = "qwen3-embedding:8b"
BATCH_SIZE = 32  # ile tekstów wysyłamy do Ollama w jednym zapytaniu /api/embed

# poziom -> (folder korpusu wejściowego, folder embeddingów wyjściowego)
LEVEL_CONFIG = {
    1: {"corpus_dir": "isco_corpus_L1", "out_dir": "isco_embeddings/level_1"},
    2: {"corpus_dir": "isco_corpus_L2", "out_dir": "isco_embeddings/level_2"},
    3: {"corpus_dir": "isco_corpus_L3", "out_dir": "isco_embeddings/level_3"},
    4: {"corpus_dir": "isco_corpus_L4", "out_dir": "isco_embeddings/level_4"},
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


def build_level(level: int, corpus_dir: str, out_dir: str, model: OllamaEmbedder):
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
        code: {"title": corpus[code]["title"], "skill_level": corpus[code]["skill_level"]}
        for code in codes_ordered
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  Zapisano embeddingi i metadane w folderze: {out_path.resolve()}")


def main():
    print(f"Model embeddingowy: {OLLAMA_MODEL} (lokalnie przez Ollama)")
    print("Upewnij się, że Ollama jest uruchomiona i masz pobrany model:")
    print(f"  ollama pull {OLLAMA_MODEL}\n")

    model = OllamaEmbedder()

    for level, cfg in LEVEL_CONFIG.items():
        build_level(level, cfg["corpus_dir"], cfg["out_dir"], model)

    print("\nGotowe - wszystkie poziomy zwektoryzowane.")


if __name__ == "__main__":
    main()
