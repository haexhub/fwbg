# Schlankes Python-Image als Basis
FROM python:3.14-slim

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

# Alle Sources kopieren (main + packages) vor der Installation. fwbg und
# fwbg-broker-ig haben zyklische Refs (broker-ig braucht fwbg>=2.0.0, fwbg
# mit [ig]-Extra braucht broker-ig), also müssen wir alles in EINER pip-
# Invocation editable installieren damit pip die Zyklen lokal auflöst.
COPY pyproject.toml .
COPY README.md* ./
COPY src/ src/
COPY packages/ packages/

# Alle editable Installs in einem Call -> pip resolved Zyklen über die
# gegebenen lokalen Paths statt PyPI zu fragen.
RUN pip install --no-cache-dir \
      -e packages/fwbg-sdk \
      -e packages/fwbg-premium \
      -e packages/fwbg-broker-ig \
      -e ".[ig,api]"

# Verzeichnisse für Volumes erstellen (damit Berechtigungen stimmen)
RUN mkdir -p accounts data logs stats_export strategies

# API Port
EXPOSE 8420

# Startbefehl
CMD ["python", "-m", "fwbg"]
