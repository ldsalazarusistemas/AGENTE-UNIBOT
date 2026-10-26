import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
import unicodedata
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from embeddings_tf import embed
from generador import generar, generar_stream

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
DOCS = Path(os.getenv("DOCS_DIR", str(BASE / "conocimiento")))
DATA = Path(os.getenv("DATA_DIR", str(BASE / "data")))
EXTENSIONES = {".md", ".txt", ".pdf", ".csv", ".json"}
INDEX_FILE = DATA / "index.json"
VEC_FILE = DATA / "vectors.npy"
INSTRUCCIONES_FILE = BASE / "instrucciones.md"

DOCS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

ENTRADAS: list[dict] = []
VECTORES = np.zeros((0, 512), dtype=np.float32)
LOCK_INDICE = threading.RLock()

MAX_FRAGMENTOS = 6
MIN_SIMILITUD = float(os.getenv("MIN_SIMILITUD", "0.20"))
RESPUESTA_SIN_EVIDENCIA = (
    "No tengo información suficiente para responder con certeza. "
    "Consulta los canales oficiales de UNINTEP."
)
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
INDEX_FORMAT_VERSION = "2"
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuales", "de", "del",
    "desde", "donde", "el", "en", "es", "esta", "este", "hay", "la",
    "las", "lo", "los", "me", "mi", "para", "por", "que", "se", "su",
    "un", "una", "uno", "y",
}
GENERIC_QUERY_TERMS = {
    "carrera", "carreras", "oferta", "ofrece", "ofrecen", "programa",
    "programas", "unin", "unintep", "universidad", "estudio", "estudiar",
}
REFUERZO_LEXICO = 0.20
REFUERZO_TERMINO_ESPECIFICO = 0.30

# --- Integración del clasificador de intenciones (Keras) -------------------
# Intención -> archivo de conocimiento que recibe un pequeño refuerzo.
INTENCION_FUENTE = {
    "programas": "programas.md",
    "admisiones": "admisiones.md",
    "biblioteca": "biblioteca.md",
    "costos": "costos.md",
    "calendario": "calendario.md",
    "bienestar": "bienestar.md",
    "practicas": "practicas_y_grados.md",
    "tramites": "tramites.md",
    "contacto": "directorio.md",
    "general": "general.md",
}
BONO_INTENCION = 0.08
CONFIANZA_REFUERZO = 0.60
CONFIANZA_DIRECTA = 0.80
RESPUESTAS_DIRECTAS = {
    "saludo": (
        "¡Hola! Soy UNIBOT, el asistente virtual de UNINTEP. "
        "¿Sobre qué tema de la institución quieres consultar?"
    ),
    "agradecimiento": (
        "¡Con gusto! Si tienes otra pregunta sobre UNINTEP, aquí estoy."
    ),
}

if not 0 <= MIN_SIMILITUD <= 1:
    raise ValueError("MIN_SIMILITUD debe estar entre 0 y 1.")


def clasificar(pregunta: str) -> dict:
    """Intención de la pregunta; nunca lanza excepciones."""
    try:
        import clasificador

        return clasificador.predecir(pregunta)
    except Exception:
        return {"intencion": "desconocida", "confianza": 0.0}


def leer(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def trocear(texto: str, max_chars: int = 900, solape: int = 150) -> list[str]:
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    bloques = []
    encabezado = ""
    for parrafo in parrafos:
        if MARKDOWN_HEADING_RE.match(parrafo):
            encabezado = "\n\n".join(
                parte for parte in (encabezado, parrafo) if parte
            )
            continue
        if encabezado:
            parrafo = f"{encabezado}\n\n{parrafo}"
            encabezado = ""
        bloques.append(parrafo)

    if encabezado:
        if bloques:
            bloques[-1] = f"{bloques[-1]}\n\n{encabezado}"
        else:
            bloques.append(encabezado)

    fragmentos = []
    actual = ""
    for parrafo in bloques:
        if len(parrafo) <= max_chars:
            candidato = (actual + "\n\n" + parrafo).strip()
            if len(candidato) <= max_chars:
                actual = candidato
                continue
        if actual:
            fragmentos.append(actual)
            actual = ""
        while len(parrafo) > max_chars:
            fragmentos.append(parrafo[:max_chars])
            parrafo = parrafo[max_chars - solape:]
        actual = parrafo
    if actual:
        fragmentos.append(actual)
    return fragmentos


def _terminos(texto: str) -> set[str]:
    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    sin_acentos = "".join(
        caracter for caracter in normalizado
        if not unicodedata.combining(caracter)
    )
    terminos = {
        termino
        for termino in TOKEN_RE.findall(sin_acentos)
        if termino not in STOPWORDS
    }
    terminos.update(
        termino[:-1]
        for termino in tuple(terminos)
        if len(termino) > 4 and re.search(r"[aeiou]s$", termino)
    )
    return terminos


def fuentes_actuales() -> dict[str, str]:
    fuentes = {}
    for archivo in sorted(DOCS.rglob("*")):
        if not archivo.is_file():
            continue
        if archivo.suffix.lower() not in EXTENSIONES:
            continue
        if any(p.startswith("_") for p in archivo.relative_to(DOCS).parts):
            continue
        try:
            # posix: la misma clave en Windows y en Docker/Linux
            ruta = archivo.relative_to(DOCS).as_posix()
            fuentes[ruta] = leer(archivo)
        except Exception as exc:
            print(f"No se pudo leer {archivo}: {exc}")
    return fuentes


def guardar_indice():
    temporal_json = INDEX_FILE.with_suffix(".tmp")
    temporal_vec = VEC_FILE.with_suffix(".tmp.npy")
    temporal_json.write_text(
        json.dumps(ENTRADAS, ensure_ascii=False), encoding="utf-8"
    )
    np.save(temporal_vec, VECTORES)
    os.replace(temporal_json, INDEX_FILE)
    os.replace(temporal_vec, VEC_FILE)


def cargar_indice():
    global ENTRADAS, VECTORES
    with LOCK_INDICE:
        if not INDEX_FILE.exists() or not VEC_FILE.exists():
            return
        entradas = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        vectores = np.load(VEC_FILE, allow_pickle=False)
        if (
            vectores.ndim == 2
            and len(entradas) == len(vectores)
            and vectores.shape[1] == 512
        ):
            ENTRADAS = entradas
            VECTORES = vectores.astype(np.float32)
        else:
            print("El índice anterior no es compatible; se reconstruirá.")
            ENTRADAS = []
            VECTORES = np.zeros((0, 512), dtype=np.float32)


def sincronizar() -> str:
    """Actualiza el índice sin recalcular documentos sin cambios."""
    global ENTRADAS, VECTORES
    with LOCK_INDICE:
        actuales = fuentes_actuales()
        hashes = {
            ruta: hashlib.sha256(
                f"{INDEX_FORMAT_VERSION}\0{texto}".encode("utf-8")
            ).hexdigest()
            for ruta, texto in actuales.items()
        }
        conservar = [
            i for i, e in enumerate(ENTRADAS)
            if hashes.get(e["fuente"]) == e["hash"]
        ]
        anteriores = ENTRADAS
        vectores_anteriores = VECTORES
        entradas = [anteriores[i] for i in conservar]
        vectores = (
            vectores_anteriores[conservar]
            if conservar
            else np.zeros((0, 512), dtype=np.float32)
        )

        indexadas = {e["fuente"] for e in entradas}
        nuevos = []
        for ruta, texto in actuales.items():
            if ruta in indexadas:
                continue
            for fragmento in trocear(texto):
                nuevos.append({
                    "fuente": ruta,
                    "hash": hashes[ruta],
                    "texto": fragmento,
                })

        for inicio in range(0, len(nuevos), 16):
            lote = nuevos[inicio:inicio + 16]
            nuevos_vec = embed([e["texto"] for e in lote])
            vectores = (
                nuevos_vec if vectores.shape[0] == 0
                else np.vstack((vectores, nuevos_vec))
            )
            entradas.extend(lote)

        # Se asigna solo al final: si algo falla, el índice previo sigue válido.
        ENTRADAS, VECTORES = entradas, vectores
        guardar_indice()
        return (
            f"Índice actualizado: {len(actuales)} fuentes, "
            f"{len(ENTRADAS)} fragmentos; "
            f"{len(nuevos)} fragmentos nuevos."
        )


def buscar_conocimiento(
    consulta: str,
    k: int = MAX_FRAGMENTOS,
    umbral: float = MIN_SIMILITUD,
    boost: dict[str, float] | None = None,
) -> list[dict]:
    with LOCK_INDICE:
        if not ENTRADAS or k <= 0:
            return []
        consulta_vector = embed([consulta])[0]
        puntuaciones = VECTORES @ consulta_vector

        terminos_consulta = _terminos(consulta)
        if terminos_consulta:
            terminos_documentos = [
                _terminos(
                    f"{entrada['fuente']} {entrada['texto']}"
                )
                for entrada in ENTRADAS
            ]
            frecuencia = {
                termino: sum(
                    termino in terminos for terminos in terminos_documentos
                )
                for termino in terminos_consulta
            }
            limite_especificidad = max(2, int(len(ENTRADAS) * 0.10))
            terminos_especificos = {
                termino
                for termino, cantidad in frecuencia.items()
                if (
                    cantidad <= limite_especificidad
                    and termino not in GENERIC_QUERY_TERMS
                )
            }
            pesos = {
                termino: np.log(
                    (len(ENTRADAS) + 1) / (frecuencia[termino] + 1)
                ) + 1
                for termino in terminos_consulta
            }
            peso_total = sum(pesos.values())
            for i, terminos in enumerate(terminos_documentos):
                cobertura = sum(
                    peso
                    for termino, peso in pesos.items()
                    if termino in terminos
                ) / peso_total
                puntuaciones[i] += REFUERZO_LEXICO * cobertura
                coincidencias_especificas = len(
                    terminos_especificos & terminos
                )
                puntuaciones[i] += min(
                    REFUERZO_TERMINO_ESPECIFICO * coincidencias_especificas,
                    0.30,
                )

        if boost:
            for i, entrada in enumerate(ENTRADAS):
                puntuaciones[i] += boost.get(entrada["fuente"], 0.0)

        candidatos = [
            int(i)
            for i in np.argsort(puntuaciones)[::-1]
            if puntuaciones[i] >= umbral
        ]
        mejores = []
        fuentes_vistas = set()
        seleccionados = set()
        limite_fuente_priorizada = min(3, k)

        if boost:
            for fuente in boost:
                agregados = 0
                for i in candidatos:
                    entrada = ENTRADAS[i]
                    fragmento = (entrada["fuente"], entrada["texto"])
                    if (
                        entrada["fuente"] != fuente
                        or fragmento in seleccionados
                    ):
                        continue
                    mejores.append(i)
                    seleccionados.add(fragmento)
                    fuentes_vistas.add(fuente)
                    agregados += 1
                    if (
                        len(mejores) == k
                        or agregados == limite_fuente_priorizada
                    ):
                        break
                if len(mejores) == k:
                    break

        for i in candidatos:
            fuente = ENTRADAS[i]["fuente"]
            fragmento = (fuente, ENTRADAS[i]["texto"])
            if fragmento in seleccionados:
                continue
            if fuente in fuentes_vistas:
                continue
            mejores.append(i)
            seleccionados.add(fragmento)
            fuentes_vistas.add(fuente)
            if len(mejores) == k:
                break

        if len(mejores) < k:
            for i in candidatos:
                fragmento = (ENTRADAS[i]["fuente"], ENTRADAS[i]["texto"])
                if fragmento in seleccionados:
                    continue
                mejores.append(i)
                seleccionados.add(fragmento)
                if len(mejores) == k:
                    break

        return [
            {
                "fuente": ENTRADAS[i]["fuente"],
                "texto": ENTRADAS[i]["texto"],
                "similitud": float(puntuaciones[i]),
            }
            for i in mejores
        ]


def construir_contexto(
    pregunta: str,
    historial: list,
    intencion: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    pregunta_anterior = next(
        (m["content"] for m in reversed(historial) if m["role"] == "user"),
        "",
    )
    consulta = pregunta
    if pregunta_anterior and len(pregunta.split()) <= 6:
        consulta = pregunta_anterior + " " + pregunta

    boost = None
    if intencion in INTENCION_FUENTE:
        boost = {INTENCION_FUENTE[intencion]: BONO_INTENCION}

    resultados = buscar_conocimiento(consulta, boost=boost)

    partes = []
    fuentes = []
    vistas = set()
    for resultado in resultados:
        partes.append(f"[Fuente: {resultado['fuente']}]\n{resultado['texto']}")
        fuente = resultado["fuente"]
        if fuente not in vistas:
            urls = list(dict.fromkeys(URL_RE.findall(resultado["texto"])))
            fuentes.append({
                "nombre": fuente,
                "url": urls[0].rstrip(".,;") if len(urls) == 1 else "",
            })
            vistas.add(fuente)
    return "\n\n".join(partes), fuentes


def cargar_instrucciones() -> str:
    if INSTRUCCIONES_FILE.exists():
        return INSTRUCCIONES_FILE.read_text(encoding="utf-8")
    return (
        "Eres UNIBOT, un asistente institucional. "
        "Responde en español, utiliza el contexto y no inventes datos."
    )


# --- Flujo común de respuesta ----------------------------------------------
def _es_cortesia(pregunta: str) -> bool:
    return len(pregunta.split()) <= 4 and "?" not in pregunta and "¿" not in pregunta


def _preparar(historial: list, pregunta: str):
    """Devuelve (respuesta_directa | None, contexto, fuentes)."""
    inicio = time.perf_counter()
    clase = clasificar(pregunta)
    logger.info(
        "chat_phase_timing fase=clasificacion duracion_ms=%.1f",
        (time.perf_counter() - inicio) * 1000,
    )
    intencion, confianza = clase["intencion"], clase["confianza"]

    if (
        intencion in RESPUESTAS_DIRECTAS
        and confianza >= CONFIANZA_DIRECTA
        and _es_cortesia(pregunta)
    ):
        return RESPUESTAS_DIRECTAS[intencion], "", []

    refuerzo = intencion if confianza >= CONFIANZA_REFUERZO else None
    inicio = time.perf_counter()
    contexto, fuentes = construir_contexto(pregunta, historial, intencion=refuerzo)
    logger.info(
        "chat_phase_timing fase=recuperacion duracion_ms=%.1f fragmentos=%d",
        (time.perf_counter() - inicio) * 1000,
        len(fuentes),
    )
    return None, contexto, fuentes


def _redactar(historial, pregunta, directa, contexto, fuentes) -> str:
    if directa is not None:
        return directa
    if not fuentes:
        return RESPUESTA_SIN_EVIDENCIA
    return generar(
        pregunta,
        contexto,
        cargar_instrucciones(),
        historial=historial,
    )


def _registrar(historial: list, pregunta: str, respuesta: str):
    historial.extend([
        {"role": "user", "content": pregunta},
        {"role": "assistant", "content": respuesta},
    ])
    historial[:] = historial[:1] + historial[1:][-12:]


def responder_con_fuentes(
    historial: list,
    pregunta: str,
) -> tuple[str, list[dict[str, str]]]:
    directa, contexto, fuentes = _preparar(historial, pregunta)
    respuesta = _redactar(historial, pregunta, directa, contexto, fuentes)
    _registrar(historial, pregunta, respuesta)
    return respuesta, fuentes


def responder(historial: list, pregunta: str) -> str:
    respuesta, _ = responder_con_fuentes(historial, pregunta)
    return respuesta


def responder_stream(historial: list, pregunta: str):
    """Emite eventos NDJSON compatibles con la interfaz web."""
    inicio_total = time.perf_counter()
    inicio_preparacion = time.perf_counter()
    directa, contexto, fuentes = _preparar(historial, pregunta)
    preparacion_ms = (time.perf_counter() - inicio_preparacion) * 1000
    yield {"t": "fuentes", "d": fuentes}

    generacion_ms = 0.0
    primer_fragmento_ms = None
    cantidad_fragmentos = 0
    estado = "ok"

    def registrar_medicion(caracteres: int):
        logger.info(
            "chat_timing estado=%s preparacion_ms=%.1f generacion_ms=%.1f "
            "primer_fragmento_ms=%s total_ms=%.1f fuentes=%d fragmentos=%d "
            "caracteres=%d",
            estado,
            preparacion_ms,
            generacion_ms,
            (
                f"{primer_fragmento_ms:.1f}"
                if primer_fragmento_ms is not None
                else "n/a"
            ),
            (time.perf_counter() - inicio_total) * 1000,
            len(fuentes),
            cantidad_fragmentos,
            caracteres,
        )

    if directa is not None:
        respuesta = directa
        yield {"t": "texto", "d": respuesta}
    elif not fuentes:
        respuesta = RESPUESTA_SIN_EVIDENCIA
        yield {"t": "texto", "d": respuesta}
    else:
        fragmentos = []
        inicio_generacion = time.perf_counter()
        try:
            for fragmento in generar_stream(
                pregunta,
                contexto,
                cargar_instrucciones(),
                historial=historial,
            ):
                if primer_fragmento_ms is None:
                    primer_fragmento_ms = (
                        time.perf_counter() - inicio_generacion
                    ) * 1000
                cantidad_fragmentos += 1
                fragmentos.append(fragmento)
                yield {"t": "texto", "d": fragmento}
        except Exception:
            estado = "error"
            raise
        finally:
            generacion_ms = (time.perf_counter() - inicio_generacion) * 1000
            if estado == "error":
                registrar_medicion(sum(map(len, fragmentos)))

        respuesta = "".join(fragmentos).strip()
        if not respuesta:
            estado = "error"
            registrar_medicion(0)
            raise RuntimeError("Ollama devolvió una respuesta vacía.")

    _registrar(historial, pregunta, respuesta)
    registrar_medicion(len(respuesta))


# --- Comandos de administración ---------------------------------------------
def aprender_texto(texto: str) -> str:
    if not texto.strip():
        raise ValueError("El texto no puede estar vacío.")
    archivo = DOCS / "aprendido.md"
    with archivo.open("a", encoding="utf-8") as salida:
        salida.write("\n\n" + texto.strip())
    return sincronizar()


def validar_url_publica(url: str):
    """Bloquea localhost e IP privadas, incluso si un dominio resuelve a ellas."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Debes proporcionar una URL HTTP o HTTPS válida.")

    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("No se permiten direcciones locales.")

    try:
        candidatas = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise ValueError("No se pudo resolver el dominio.") from exc
    for candidata in candidatas:
        if not ip_address(candidata).is_global:
            raise ValueError("No se permiten IP privadas o reservadas.")
    return parsed


def aprender_web(url: str) -> str:
    validar_url_publica(url)
    respuesta = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "UNIBOT/1.0"},
        allow_redirects=False,
    )
    respuesta.raise_for_status()

    soup = BeautifulSoup(respuesta.text, "html.parser")
    for etiqueta in soup(
        ["script", "style", "nav", "footer", "header", "noscript"]
    ):
        etiqueta.decompose()
    texto = soup.get_text("\n\n", strip=True)

    nombre = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20] + ".md"
    carpeta = DOCS / "web"
    carpeta.mkdir(exist_ok=True)
    (carpeta / nombre).write_text(f"Fuente: {url}\n\n{texto}", encoding="utf-8")
    return sincronizar()


def listar_fuentes() -> str:
    with LOCK_INDICE:
        conteo = {}
        for entrada in ENTRADAS:
            conteo[entrada["fuente"]] = conteo.get(entrada["fuente"], 0) + 1
    if not conteo:
        return "Todavía no hay documentos indexados."
    return "Fuentes indexadas:\n" + "\n".join(
        f"- {fuente}: {cantidad} fragmentos"
        for fuente, cantidad in sorted(conteo.items())
    )


def ejecutar_comando(texto: str, comandos_activos: bool = True) -> str:
    orden, _, argumento = texto.strip().partition(" ")
    orden = orden.lower()
    argumento = argumento.strip()

    if orden == "/ayuda":
        return (
            "/ayuda — muestra los comandos\n"
            "/fuentes — lista los documentos indexados\n"
            "/recargar — actualiza el índice\n"
            "/aprender texto — añade conocimiento\n"
            "/web URL — descarga una página pública"
        )
    if orden == "/fuentes":
        return listar_fuentes()
    if not comandos_activos:
        return "Los comandos de modificación están desactivados."
    if orden == "/recargar":
        return sincronizar()
    if orden == "/aprender":
        if not argumento:
            return "Uso: /aprender texto"
        return aprender_texto(argumento)
    if orden == "/web":
        if not argumento:
            return "Uso: /web https://ejemplo.com/"
        return aprender_web(argumento)
    return "Comando desconocido. Escribe /ayuda."