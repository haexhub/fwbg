# Schlankes Python-Image als Basis
FROM python:3.14-slim

# Verhindert, dass Python .pyc Dateien schreibt und sorgt für sofortige Log-Ausgabe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System-Abhängigkeiten installieren
# libgomp1 ist zwingend erforderlich für XGBoost!
# git: das Research-Register (Bind-Mount) wird per git geschrieben.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
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
      -e packages/fwbg-broker-ig \
      -e ".[ig,api]"

# Unprivilegierter Benutzer mit fester UID/GID (Volumes muessen dieser UID
# gehoeren, siehe docs/deployment.md).
RUN groupadd --gid 10001 fwbg \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/fwbg \
       --shell /usr/sbin/nologin fwbg

# Verzeichnisse für Volumes erstellen (damit Berechtigungen stimmen). Ein
# leeres Named Volume uebernimmt Eigentuemer und Rechte des Image-Verzeichnisses
# beim ersten Mount; deshalb gehoeren alle Mountpunkte dem Benutzer fwbg.
# /home/fwbg/fwbg ist der Workspace, /home/fwbg/.fwbg/plugins das Plugin-Volume.
RUN mkdir -p accounts data/stats_export logs stats_export strategies \
      /home/fwbg/fwbg /home/fwbg/.fwbg/plugins \
    && chown fwbg:fwbg /app \
    && chown -R fwbg:fwbg accounts data logs stats_export strategies /home/fwbg

# Das Research-Register ist ein Bind-Mount vom Host und gehoert dort einem
# anderen Benutzer; ohne safe.directory verweigert git die Arbeit darin.
RUN git config --system --add safe.directory /home/fwbg/fwbg/research

ENV HOME=/home/fwbg
USER fwbg:fwbg

# API Port
EXPOSE 8420

# Startbefehl
CMD ["python", "-m", "fwbg"]
