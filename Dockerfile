# Utilisation d'une image Python légère et officielle
FROM python:3.11-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Optimisations Python (ne pas écrire de fichiers .pyc, et forcer l'affichage des logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installation des dépendances système de base
# (Parfois nécessaires pour compiler certaines dépendances de PyMuPDF ou lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie uniquement le fichier des dépendances en premier
# (optimise le cache de build Docker)
COPY requirements.txt .

# Installation des paquets Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code source
COPY . .

# Exposition du port utilisé par FastAPI
EXPOSE 8000

# Commande de lancement du serveur uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]