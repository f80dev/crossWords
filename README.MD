# CrossWords - Générateur de Mots Croisés

API FastAPI pour générer des grilles de mots croisés à partir de documents PDF ou EPUB.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
uvicorn main:app --reload
```

## Endpoints

- `POST /api/v1/generate` — Upload un PDF/EPUB → reçoit un PDF de grille
- `POST /api/v1/export/pdf` — Reçoit du JSON → retourne un PDF

## Environment

- `MINIMAX_API_KEY` — Clé API MiniMax (optionnel)