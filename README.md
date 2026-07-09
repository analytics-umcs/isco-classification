# ISCO-08 Classification App

Streamlit application for assisted ISCO-08 occupation classification using European Social Survey fields and local Ollama embeddings.

## Local Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Pull the local embedding model:

```bash
ollama pull qwen3-embedding:8b
```

Run the app:

```bash
python3 -m streamlit run app.py
```

The app requires a running Ollama service and the `qwen3-embedding:8b` model.

## Users

The app includes a simple local Streamlit login gate with users `User1` through `User10`.

## Deployment Files

Main Streamlit file:

```text
app.py
```

The app bundles required ISCO embedding files under `isco_embeddings/` and logo assets under `assets/`.

## GitHub Pages

GitHub Pages can only serve static files. It cannot run this Streamlit/Ollama application.

The project information page is intended for:

`https://analytics-umcs.github.io/isco-classification/`

To run the actual app publicly, deploy it to a Python-capable host with Ollama access, such as a VM, internal server, or Streamlit-compatible platform. A hosted environment without Ollama can render the interface, but model-backed classification will fail until Ollama is provided.
