# Assignment 1: Coding Environment Setup and Simple API Implementation

I've developed Assignment 1 by extending Module 3's activity to add a **/embedding** endpoint using **spaCy** to view the word embeddings (from Module 2). 

A **Dockerfile** is included so anyone can run the app in a clean, reproducible environment.

---

## Project Layout

```
sps_genai/
├─ app/
│  ├─ main.py            # FastAPI routes: "/", /generate, /embedding
│  └─ bigram_model.py    # Bigram model + spaCy embedding 
├─ Dockerfile
├─ pyproject.toml
└─ uv.lock
```

---

## Running with Docker

After cloning this repo, build and run the image from the sps_genai directory:

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

To open on your device:
- Swagger UI: http://127.0.0.1:8000/docs
- Root:       http://127.0.0.1:8000/

> The container listens on port **80**; `-p 8000:80` maps it to host port **8000**.

---

## Local Run (without Docker)

**Requirements:** [uv](https://github.com/astral-sh/uv) (or Python 3.12+)

**NOTE:** Ensure you run this part from the sps_genai directory.

```bash
# install dependencies into .venv
uv sync

# install a spaCy English model with vectors (pick ONE)
uv run python -m spacy download en_core_web_lg
# or:
uv run python -m spacy download en_core_web_md

# start the server
uv run fastapi dev app/main.py
```

Open: http://127.0.0.1:8000/docs

---

## API

### POST `/generate`
Generate text from a **bigram** model trained on a small in-file corpus (defined in `app/main.py`).

**Example Request**
```json
{
  "start_word": "the",
  "length": 12
}
```

**Response**
```json
{
  "generated_text": "count of monte cristo is a novel ..."
}
```

Notes:
- If the start word is unseen or has no followers, the model **backs off** to the most similar in-vocab word using spaCy embeddings, then continues generation.
- The model only produces words present in the corpus vocabulary.

---

### POST `/embedding`
Return the **spaCy embedding** vector for a single word.

**Example Request**
```json
{ "word": "data" }
```

**Response**
```json
{
  "word": "data",
  "dim": 300,
  "vector": [ ... floats ... ]
}
```

---

## How it works

- `BigramModel` tokenizes the **corpus** (list of sentences in `main.py`), builds **bigram counts**, and samples next words with **Laplace smoothing**.
- If the chosen start word is unseen/has no outgoing bigrams, it **backs off** to the most embedding-similar known word (via spaCy) and resumes generation.
- `/embedding` returns `nlp(word).vector` as JSON (dimension depends on the installed spaCy model, e.g., 300 for `lg`).
---