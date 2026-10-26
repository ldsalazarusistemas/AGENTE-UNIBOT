import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import app as bot
from generador import ErrorGenerador

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
WEB = BASE / "web"

ESTADO = {
    "listo": False,
    "mensaje": "Iniciando TensorFlow y preparando documentos...",
    "fragmentos": 0,
    "advertencia": "",
}
SESIONES: OrderedDict = OrderedDict()
LOCK = threading.RLock()
LOCK_GENERACION = threading.Lock()
MAX_SESIONES = 200

COMANDOS_ACTIVOS = os.getenv("COMANDOS_ADMIN", "0") == "1"
COMANDOS_ADMIN_TOKEN = os.getenv("COMANDOS_ADMIN_TOKEN", "")
COMANDOS_LECTURA = {"/ayuda", "/fuentes"}
PRECARGAR_GENERADOR = os.getenv("PRECARGAR_GENERADOR", "1") == "1"


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class ChatJsonIn(BaseModel):
    pregunta: str = Field(min_length=1, max_length=1000)


class ChatJsonOut(BaseModel):
    respuesta: str
    fuentes: list[dict[str, str]]


def linea(evento: dict) -> str:
    return json.dumps(evento, ensure_ascii=False) + "\n"


def preparar_clasificador():
    """Entrena (si hace falta) y carga el clasificador de intenciones Keras."""
    import clasificador

    if not clasificador.RUTA_MODELO.exists() or not clasificador.RUTA_CLASES.exists():
        import entrenar

        ESTADO["mensaje"] = "Entrenando clasificador de intenciones (Keras)..."
        entrenar.entrenar()
    clasificador.cargar()


def preparar_aplicacion():
    try:
        try:
            preparar_clasificador()
        except Exception:
            logger.exception("Clasificador no disponible; se continúa sin él")

        ESTADO["mensaje"] = "Cargando índice documental..."
        inicio = time.perf_counter()
        bot.cargar_indice()
        logger.info(
            "startup_timing fase=cargar_indice duracion_ms=%.1f fragmentos=%d",
            (time.perf_counter() - inicio) * 1000,
            len(bot.ENTRADAS),
        )

        ESTADO["mensaje"] = "Cargando el modelo de embeddings (TensorFlow Hub)..."
        inicio = time.perf_counter()
        bot.embed(["Inicialización de UNIBOT"])
        logger.info(
            "startup_timing fase=calentar_embeddings duracion_ms=%.1f",
            (time.perf_counter() - inicio) * 1000,
        )

        ESTADO["mensaje"] = "Indexando documentos..."
        inicio = time.perf_counter()
        resultado = bot.sincronizar()
        logger.info(
            "startup_timing fase=sincronizar_documentos duracion_ms=%.1f "
            "resultado=%s",
            (time.perf_counter() - inicio) * 1000,
            resultado,
        )

        if PRECARGAR_GENERADOR:
            ESTADO["mensaje"] = "Preparando el modelo local de Ollama; puede tardar..."
            try:
                import generador

                inicio = time.perf_counter()
                generador.cargar_modelo()
                logger.info(
                    "startup_timing fase=preparar_ollama duracion_ms=%.1f",
                    (time.perf_counter() - inicio) * 1000,
                )
            except ErrorGenerador as exc:
                logger.error("Modelo de Ollama no disponible: %s", exc)
                ESTADO["advertencia"] = str(exc)
            except Exception as exc:
                logger.exception("No se pudo preparar el modelo de Ollama")
                ESTADO["advertencia"] = f"No se pudo preparar Ollama: {exc}"

        ESTADO.update(
            listo=True,
            mensaje="UNIBOT está en línea",
            fragmentos=len(bot.ENTRADAS),
        )
    except Exception as exc:
        ESTADO.update(
            listo=False,
            mensaje=(
                "No se pudo preparar el RAG: "
                f"{type(exc).__name__}: {exc}. Revisa los registros."
            ),
        )
        logger.exception("Error al preparar el RAG")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=preparar_aplicacion, daemon=True).start()
    yield


api = FastAPI(title="UNIBOT TensorFlow", version="1.1.0", lifespan=lifespan)

origenes_cors = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]
api.add_middleware(
    CORSMiddleware,
    allow_origins=origenes_cors,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


def historial_de(session_id: str) -> list:
    with LOCK:
        if session_id not in SESIONES:
            SESIONES[session_id] = [
                {"role": "system", "content": bot.cargar_instrucciones()}
            ]
        SESIONES.move_to_end(session_id)
        while len(SESIONES) > MAX_SESIONES:
            SESIONES.popitem(last=False)
        return SESIONES[session_id]


def token_valido(token: str | None) -> bool:
    return bool(COMANDOS_ADMIN_TOKEN) and secrets.compare_digest(
        token or "", COMANDOS_ADMIN_TOKEN
    )


@api.get("/api/estado")
def estado():
    return dict(ESTADO)


@api.get("/health")
def health():
    status_code = 200 if ESTADO["listo"] else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if ESTADO["listo"] else "starting", **ESTADO},
    )


@api.post("/chat", response_model=ChatJsonOut)
def chat_json(body: ChatJsonIn):
    if not ESTADO["listo"]:
        return JSONResponse(status_code=503, content={"detail": ESTADO["mensaje"]})

    pregunta = body.pregunta.strip()
    if not pregunta:
        return JSONResponse(
            status_code=422, content={"detail": "Escribe una pregunta."}
        )

    historial = [{"role": "system", "content": bot.cargar_instrucciones()}]
    try:
        with LOCK_GENERACION:
            respuesta, fuentes = bot.responder_con_fuentes(historial, pregunta)
    except ErrorGenerador as exc:
        logger.error("No se pudo acceder al modelo generativo: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return ChatJsonOut(respuesta=respuesta, fuentes=fuentes)


@api.post("/api/chat")
def chat(
    body: ChatIn,
    # El alias debe coincidir con el encabezado que permite CORS.
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    cabeceras = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    def respuesta_error(texto: str):
        return StreamingResponse(
            iter([linea({"t": "error", "d": texto}), linea({"t": "fin"})]),
            media_type="application/x-ndjson",
            headers=cabeceras,
        )

    if not ESTADO["listo"]:
        return respuesta_error(ESTADO["mensaje"])

    mensaje = body.message.strip()
    if not mensaje:
        return respuesta_error("Escribe una pregunta.")

    if mensaje.startswith("/"):
        def comando():
            try:
                orden = mensaje.partition(" ")[0].lower()
                es_lectura = orden in COMANDOS_LECTURA

                if not es_lectura and COMANDOS_ACTIVOS:
                    if not COMANDOS_ADMIN_TOKEN:
                        yield linea({"t": "error", "d": (
                            "Los comandos administrativos no están "
                            "disponibles: falta configurar su token."
                        )})
                        yield linea({"t": "fin"})
                        return
                    if not token_valido(x_admin_token):
                        yield linea({"t": "error", "d": (
                            "No autorizado para ejecutar este comando."
                        )})
                        yield linea({"t": "fin"})
                        return

                texto = bot.ejecutar_comando(
                    mensaje,
                    comandos_activos=COMANDOS_ACTIVOS
                    and (es_lectura or token_valido(x_admin_token)),
                )
                yield linea({"t": "texto", "d": texto})
            except Exception:
                logger.exception("Error al ejecutar comando %r", mensaje)
                yield linea({"t": "error", "d": "No se pudo ejecutar el comando."})
            yield linea({"t": "fin"})

        return StreamingResponse(
            comando(), media_type="application/x-ndjson", headers=cabeceras
        )

    historial = historial_de(body.session_id)

    def generar_eventos():
        try:
            with LOCK_GENERACION:
                for evento in bot.responder_stream(historial, mensaje):
                    yield linea(evento)
        except ErrorGenerador as exc:
            logger.error("No se pudo acceder al modelo generativo: %s", exc)
            yield linea({"t": "error", "d": str(exc)})
        except Exception as exc:
            logger.exception("Error al responder una pregunta")
            yield linea({"t": "error", "d": (
                f"No pude generar la respuesta ({type(exc).__name__}). "
                "Revisa los registros del servidor y la memoria disponible."
            )})
        yield linea({"t": "fin"})

    return StreamingResponse(
        generar_eventos(), media_type="application/x-ndjson", headers=cabeceras
    )


if WEB.exists():
    api.mount("/", StaticFiles(directory=str(WEB), html=True))