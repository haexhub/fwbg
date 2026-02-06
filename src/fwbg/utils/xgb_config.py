"""
XGBoost Konfiguration - Zentrale Steuerung von n_jobs und GPU.

Wird von process.py gesetzt basierend auf der Anzahl paralleler Feature-Gruppen
um Überparallelisierung zu vermeiden.

GPU-Unterstützung:
- Automatische Erkennung von CUDA-fähigen GPUs
- Fallback auf CPU wenn keine GPU verfügbar
- Kompatibel mit NVIDIA, AMD (via ROCm), und CPU-only Systemen
"""
import os
import logging

logger = logging.getLogger(__name__)

# Globale Variablen
_XGBOOST_N_JOBS = -1
_GPU_AVAILABLE = None  # Cache für GPU-Check
_GPU_DEVICE_ID = 0


def _check_gpu_available() -> bool:
    """
    Prüft ob eine CUDA-fähige GPU für XGBoost verfügbar ist.

    Returns:
        True wenn GPU nutzbar, False sonst
    """
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Umgebungsvariable zum Deaktivieren der GPU
    if os.environ.get("FWBG_NO_GPU", "").lower() in ("1", "true", "yes"):
        logger.info("GPU deaktiviert via FWBG_NO_GPU Umgebungsvariable")
        _GPU_AVAILABLE = False
        return False

    try:
        # Versuche XGBoost mit GPU zu initialisieren
        import xgboost as xgb

        # Prüfe ob XGBoost mit GPU-Support kompiliert wurde
        # und ob eine GPU verfügbar ist
        # XGBoost 2.0+: device='cuda' mit tree_method='hist'
        test_params = {
            "tree_method": "hist",
            "device": "cuda",
            "n_estimators": 1,
            "max_depth": 1,
            "verbosity": 0,
        }

        # Kleiner Test-Datensatz
        import numpy as np
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 0, 1])

        model = xgb.XGBClassifier(**test_params)
        model.fit(X_test, y_test)

        _GPU_AVAILABLE = True
        logger.info("GPU erkannt und für XGBoost aktiviert (device=cuda)")
        return True

    except Exception as e:
        _GPU_AVAILABLE = False
        logger.debug(f"GPU nicht verfügbar: {e}")
        return False


def set_xgboost_n_jobs(n_jobs: int):
    """Setzt n_jobs für alle XGBoost-Modelle."""
    global _XGBOOST_N_JOBS
    _XGBOOST_N_JOBS = n_jobs


def get_xgboost_n_jobs() -> int:
    """Gibt aktuelles n_jobs für XGBoost zurück."""
    return _XGBOOST_N_JOBS


def set_gpu_device(device_id: int):
    """Setzt die GPU-Device-ID (für Multi-GPU Systeme)."""
    global _GPU_DEVICE_ID
    _GPU_DEVICE_ID = device_id


def get_xgboost_params() -> dict:
    """
    Gibt optimale XGBoost-Parameter zurück (GPU wenn verfügbar).

    Returns:
        Dict mit tree_method und device Parametern
    """
    if _check_gpu_available():
        return {
            "tree_method": "hist",
            "device": f"cuda:{_GPU_DEVICE_ID}",
        }
    else:
        return {
            "tree_method": "hist",
            "device": "cpu",
        }


def is_gpu_available() -> bool:
    """Prüft ob GPU für XGBoost verfügbar ist."""
    return _check_gpu_available()
