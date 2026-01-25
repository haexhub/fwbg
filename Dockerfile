# Schlankes Python-Image als Basis
FROM python:3.10-slim

# Verhindert, dass Python .pyc Dateien schreibt und sorgt für sofortige Log-Ausgabe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System-Abhängigkeiten installieren
# libgomp1 ist zwingend erforderlich für XGBoost!
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

# Verzeichnisse für Volumes erstellen (damit Berechtigungen stimmen)
RUN mkdir -p accounts data logs stats_export

# Startbefehl (Bot ausführen)
CMD ["python", "ig_bot.py"]
