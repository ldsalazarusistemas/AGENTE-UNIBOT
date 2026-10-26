import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
# Registers the custom op required by the multilingual USE SavedModel.
import tensorflow_text
import tensorflow_hub as hub

URL_EMBEDDINGS = (
    "https://tfhub.dev/google/"
    "universal-sentence-encoder-multilingual/3"
)

_modelo = None


def cargar_modelo():
    """Carga el modelo una sola vez por proceso."""
    global _modelo

    if _modelo is None:
        _modelo = hub.load(URL_EMBEDDINGS)

    return _modelo


def embed(textos: list[str], lote: int = 16) -> np.ndarray:
    """Convierte una lista de textos en vectores normalizados."""
    if not textos:
        return np.zeros((0, 512), dtype=np.float32)

    modelo = cargar_modelo()
    resultados = []

    for inicio in range(0, len(textos), lote):
        grupo = textos[inicio:inicio + lote]
        vectores = modelo(tf.constant(grupo))
        vectores = tf.math.l2_normalize(vectores, axis=1)
        resultados.append(vectores.numpy().astype(np.float32))

    return np.vstack(resultados)


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcula similitud entre vectores ya normalizados."""
    return np.asarray(a) @ np.asarray(b).T