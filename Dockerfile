# Schlankes Python-Image als Basis
FROM python:3.11-slim

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

# pyproject.toml und src kopieren für Installation
COPY pyproject.toml .
COPY src/ src/

# Package mit IG-Dependencies installieren
RUN pip install --no-cache-dir -e ".[ig]"

# Verzeichnisse für Volumes erstellen (damit Berechtigungen stimmen)
RUN mkdir -p accounts data logs stats_export

# Startbefehl
CMD ["python", "-m", "fwbg"]
