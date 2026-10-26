import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELOS = BASE / "modelos"
RUTA_DATOS = DATA / "intenciones.json"


def entrenar():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    DATA.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)

    if not RUTA_DATOS.exists():
        import preparar_datos

        preparar_datos.guardar_ejemplos()

    datos = json.loads(RUTA_DATOS.read_text(encoding="utf-8"))
    textos = [item["texto"] for item in datos]
    clases = sorted({item["intencion"] for item in datos})
    mapa = {clase: i for i, clase in enumerate(clases)}
    etiquetas = np.array([mapa[item["intencion"]] for item in datos])

    vectorizador = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=30,
        standardize="lower_and_strip_punctuation",
    )
    vectorizador.adapt(tf.data.Dataset.from_tensor_slices(textos).batch(16))

    entrada = keras.Input(shape=(), dtype="string", name="pregunta")
    x = vectorizador(entrada)
    x = layers.Embedding(5000, 64, mask_zero=True)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    salida = layers.Dense(len(clases), activation="softmax")(x)

    modelo = keras.Model(entrada, salida)
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(
        tf.constant(textos, dtype=tf.string),
        etiquetas,
        epochs=80,
        batch_size=8,
        verbose=2,
    )

    modelo.save(MODELOS / "clasificador.keras")
    (MODELOS / "clases.json").write_text(
        json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Clasificador entrenado ({len(clases)} intenciones) y guardado.")


if __name__ == "__main__":
    entrenar()