"""Clasificador de intenciones (Keras / TensorFlow).

Las dependencias pesadas se importan de forma diferida para que el módulo
sea barato de importar y nunca tumbe el servidor si el modelo no existe.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
MODELOS = BASE / "modelos"
RUTA_MODELO = MODELOS / "clasificador.keras"
RUTA_CLASES = MODELOS / "clases.json"
UMBRAL = float(os.getenv("UMBRAL_INTENCION", "0.50"))

_modelo = None
_clases: list[str] = []
_fallo = False
_lock = threading.Lock()

DESCONOCIDA = {"intencion": "desconocida", "confianza": 0.0}


def cargar() -> bool:
    global _modelo, _clases, _fallo
    with _lock:
        if _modelo is not None:
            return True
        if _fallo or not RUTA_MODELO.exists() or not RUTA_CLASES.exists():
            return False
        try:
            import keras

            _modelo = keras.models.load_model(RUTA_MODELO)
            _clases = json.loads(RUTA_CLASES.read_text(encoding="utf-8"))
            return True
        except Exception:
            logger.exception("No se pudo cargar el clasificador de intenciones")
            _fallo = True
            return False


def predecir(texto: str) -> dict:
    if not cargar():
        return dict(DESCONOCIDA)
    try:
        import tensorflow as tf

        prob = _modelo(tf.constant([texto]), training=False).numpy()[0]
        indice = int(prob.argmax())
        confianza = float(prob[indice])
        if confianza < UMBRAL:
            return {"intencion": "desconocida", "confianza": confianza}
        return {"intencion": _clases[indice], "confianza": confianza}
    except Exception:
        logger.exception("Error al clasificar la intención")
        return dict(DESCONOCIDA)