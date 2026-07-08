# ISCO Classification

Streamlit application for assisted ISCO-08 occupation classification based on European Social Survey data.

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

The app requires a running local Ollama service and the `qwen3-embedding:8b` model.

## Users

The app includes a simple local Streamlit login gate with users `User1` through `User10`.

## GitHub Pages

GitHub Pages can only serve static files. It cannot run this Streamlit/Ollama application.

The project information page is intended for:

`https://analytics-umcs.github.io/isco-classification/`

To run the actual app publicly, deploy it to a Python-capable host with Ollama access, such as a VM, internal server, or Streamlit-compatible platform.
