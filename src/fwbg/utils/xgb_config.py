"""
XGBoost Konfiguration - Zentrale Steuerung von n_jobs.

Wird von process.py gesetzt basierend auf der Anzahl paralleler Feature-Gruppen
um Überparallelisierung zu vermeiden.
"""

# Globale Variable für XGBoost n_jobs
# Default: -1 (alle Kerne) - wird überschrieben wenn Feature-Gruppen parallel laufen
_XGBOOST_N_JOBS = -1


def set_xgboost_n_jobs(n_jobs: int):
    """Setzt n_jobs für alle XGBoost-Modelle."""
    global _XGBOOST_N_JOBS
    _XGBOOST_N_JOBS = n_jobs


def get_xgboost_n_jobs() -> int:
    """Gibt aktuelles n_jobs für XGBoost zurück."""
    return _XGBOOST_N_JOBS
