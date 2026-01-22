# Schlankes Python-Image als Basis
FROM python:3.10-slim

# System-Abhängigkeiten für XGBoost und Pandas installieren
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Den restlichen Code kopieren
COPY . .

# Ordner für persistente Daten erstellen
RUN mkdir -p data stats_export

# Startbefehl (Bot ausführen)
CMD ["python", "ig_bot.py"]
