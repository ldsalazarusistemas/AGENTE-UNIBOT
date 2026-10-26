# UNIBOT

Asistente institucional en español para consultar documentos locales. Usa
recuperación aumentada por generación (RAG): TensorFlow transforma preguntas
y documentos en vectores para buscarlos, Keras clasifica el tema de la
pregunta y Ollama redacta una respuesta con los fragmentos recuperados.

## Requisitos

- Windows con Docker Desktop instalado y abierto.
- Docker Compose v2.
- Conexión a Internet en el primer inicio para descargar imágenes y modelos.
- Memoria disponible para los contenedores. UNIBOT tiene un límite de 7 GB
  configurado en Docker Compose.

No es necesario instalar Python, TensorFlow, Keras u Ollama directamente en
Windows. Se ejecutan en contenedores.

## Iniciar y detener

Abre PowerShell en la carpeta del proyecto:

```powershell
Set-Location "C:\Users\CASAG\Desktop\PROYECTO IA\asistentetensorflow"
```

Construye la imagen de UNIBOT:

```powershell
docker compose build
```

Inicia UNIBOT y Ollama en segundo plano:

```powershell
docker compose up -d
```

Abre [http://localhost:8000](http://localhost:8000). La primera preparación
puede tardar: descarga el modelo de embeddings, comprueba o entrena el
clasificador Keras, indexa los documentos y prepara el modelo de Ollama.
Consulta el estado en [http://localhost:8000/health](http://localhost:8000/health);
la aplicación está lista cuando devuelve `"status": "ok"`.

Comandos cotidianos:

```powershell
docker compose ps                 # Ver estado de los servicios
docker compose logs -f unibot     # Ver preparación y errores de UNIBOT
docker compose logs -f ollama     # Ver el servicio Ollama
docker compose restart unibot     # Reiniciar UNIBOT y volver a sincronizar documentos
docker compose down               # Detener los servicios
docker compose up -d              # Volver a iniciarlos
```

### Revisar dónde se demora

Los registros de UNIBOT separan la sincronización inicial de los documentos
y los tiempos de cada consulta. Para ver las últimas mediciones en PowerShell:

```powershell
docker compose logs --since 30m unibot |
  Select-String "startup_timing|chat_timing"
```

`startup_timing fase=sincronizar_documentos` mide la lectura e indexación de
los documentos al iniciar. En cada `chat_timing`, `preparacion_ms` mide la
clasificación y búsqueda de información. Los registros
`chat_phase_timing` separan esas fases: `clasificacion` y `recuperacion`.
`generacion_ms` mide la transmisión desde Ollama; y `primer_fragmento_ms`
indica cuánto esperó hasta el primer fragmento generado. Si la recuperación
domina, hay que optimizar el índice; si domina la generación o el primer
fragmento, hay que revisar el modelo, la carga de Ollama o el tamaño del
prompt.

Para reconstruir después de cambiar código o dependencias:

```powershell
docker compose up -d --build
```

## Puertos y servicios

| Servicio | Dirección dentro de Docker | Acceso desde Windows |
| --- | --- | --- |
| UNIBOT: interfaz y API | `unibot:8000` | `http://localhost:8000` |
| Ollama: API de modelos | `ollama:11434` | No se publica en el host por defecto |

El puerto `8000` está enlazado a `127.0.0.1`: solo es accesible desde el mismo
equipo, no desde otros dispositivos de la red. Ollama escucha dentro de la red
de Docker y UNIBOT se conecta a `http://ollama:11434`. No hace falta abrir ni
publicar el puerto `11434` para el uso normal.

## Cómo funciona

### Preparación de documentos

1. UNIBOT lee archivos compatibles de `conocimiento/`: Markdown, texto, PDF,
   CSV y JSON.
2. Divide su contenido en fragmentos y conserva los encabezados de Markdown
   junto con el texto relacionado.
3. TensorFlow Hub genera un vector de 512 dimensiones por fragmento.
4. Guarda textos y metadatos en `data/index.json`, y vectores en
   `data/vectors.npy`. Los archivos que no cambiaron pueden reutilizar sus
   vectores existentes.

### Respuesta a una pregunta

1. La interfaz envía la pregunta a la API de UNIBOT.
2. Keras estima la intención (por ejemplo, programas, biblioteca o costos).
   Si la confianza es suficiente, la intención ayuda a priorizar el documento
   asociado; no contiene las respuestas institucionales.
3. TensorFlow genera el vector de la pregunta. El buscador combina similitud
   semántica con coincidencias de palabras, y selecciona hasta seis
   fragmentos relevantes. Prioriza fuentes distintas y puede añadir otros
   fragmentos cuando son útiles.
4. Ollama recibe la pregunta, el historial reciente y los fragmentos
   recuperados, y transmite la respuesta a medida que la genera. Si no hay
   evidencia suficiente, el programa informa que no puede responder con
   certeza en lugar de llamar al generador.
5. La API devuelve la respuesta y las fuentes para que se pueda comprobar qué
   documentos se usaron.

Los tres componentes tienen responsabilidades distintas: TensorFlow hace la
representación y búsqueda semántica; Keras clasifica la intención; Ollama
genera el texto de respuesta.

## Archivos y persistencia

- `conocimiento/`: documentos consultados. Editar o añadir un documento no
  requiere reconstruir la imagen; reinicia UNIBOT para que sincronice el índice.
- `data/`: índice de textos y vectores, además de los ejemplos de intenciones.
- `modelos/`: clasificador Keras y caché del modelo de embeddings.
- `ollama_data`: volumen Docker con el modelo de lenguaje descargado.
- `web/`: interfaz del chat.

Los datos de conocimiento, índice y modelos se conservan al detener los
contenedores con `docker compose down`. No uses `docker compose down -v` si
quieres conservar los volúmenes de Docker, como `ollama_data`.

## Actualizar el conocimiento

1. Añade o edita documentos dentro de `conocimiento/`.
2. Usa fuentes oficiales e incluye URLs. Para datos temporales, anota su fecha
   de consulta y vigencia.
3. Reinicia el servicio para sincronizar:

   ```powershell
   docker compose restart unibot
   ```

4. Confirma que volvió a estar listo:

   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   ```

Los documentos son la fuente de las respuestas. Los ejemplos de
`data/intenciones.json` y el script `preparar_datos.py` sirven para entrenar
el clasificador Keras; añadir una frase de entrenamiento no agrega datos de
conocimiento.

## API

### Chat simple

`POST /chat` acepta una pregunta y devuelve una respuesta completa:

```json
{
  "pregunta": "¿Qué programas ofrece UNINTEP?"
}
```

Respuesta:

```json
{
  "respuesta": "Respuesta basada en los documentos recuperados.",
  "fuentes": [
    {
      "nombre": "programas.md",
      "url": "https://unintep.edu.co/portal/programas-academicos/"
    }
  ]
}
```

Ejemplo desde PowerShell:

```powershell
$body = @{ pregunta = "Que programas ofrece UNINTEP?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post `
  -ContentType "application/json" -Body $body
```

### Chat con historial

`POST /api/chat` acepta `session_id` y `message`. La respuesta es NDJSON:
incluye un evento de fuentes, eventos de texto conforme Ollama los va
generando y un evento final. La interfaz web usa este endpoint. El historial
se guarda en memoria del servidor, se limita a 200 sesiones y se pierde al
reiniciar UNIBOT.

Otros endpoints:

- `GET /health`: devuelve HTTP 503 mientras prepara la aplicación y HTTP 200
  cuando está lista.
- `GET /api/estado`: muestra estado, mensaje de preparación, cantidad de
  fragmentos y posibles advertencias.

## Configuración

Se pueden establecer variables en un archivo `.env` junto a
`docker-compose.yml`. Los valores predeterminados actuales son:

| Variable | Predeterminado | Uso |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo que Ollama comprueba y descarga si falta |
| `MIN_SIMILITUD` | `0.20` | Umbral de puntuación para incluir fragmentos |
| `MAX_TOKENS` | `384` | Límite de tokens de la respuesta de Ollama |
| `MAX_PROMPT_CHARS` | `12000` | Límite aproximado del contexto del prompt |
| `OLLAMA_TIMEOUT` | `1800` | Tiempo máximo de espera, en segundos |
| `PRECARGAR_GENERADOR` | `1` | Comprueba Ollama durante el arranque |
| `CORS_ORIGINS` | `*` | Orígenes web permitidos, separados por comas |
| `COMANDOS_ADMIN` | `0` | Activa o desactiva comandos que modifican datos |
| `COMANDOS_ADMIN_TOKEN` | vacío | Token requerido para esos comandos |

Ejemplo para comenzar un `.env`:

```powershell
if (!(Test-Path .env)) { New-Item .env -ItemType File }
```

Después de cambiar variables, recrea el contenedor para aplicarlas:

```powershell
docker compose up -d --force-recreate
```

`MIN_SIMILITUD` afecta la cantidad de fragmentos recuperados: si es muy alto,
puede excluir evidencia válida; si es muy bajo, puede incorporar fragmentos
menos relacionados. El ranking también incluye refuerzos semánticos y
léxicos, así que el valor final no es solo similitud coseno.

## Pruebas y evaluación

Ejecuta las pruebas unitarias:

```powershell
docker compose run --rm unibot python -m unittest discover -s tests -v
```

Evalúa que las preguntas de `tests/evaluacion.json` recuperen la fuente
esperada:

```powershell
docker compose run --rm unibot python evaluar_recuperacion.py
```

La evaluación sincroniza los documentos con el índice, usa embeddings reales
de TensorFlow y no llama a Ollama. Mide si la fuente esperada aparece entre
los resultados; no garantiza por sí sola que el texto generado sea perfecto.

## Comandos administrativos

Los comandos de solo lectura `/ayuda` y `/fuentes` están disponibles en el
chat. Los comandos que cambian documentos están desactivados por defecto.
Para habilitarlos, configura `COMANDOS_ADMIN=1` y un
`COMANDOS_ADMIN_TOKEN` largo y secreto en `.env`. Los comandos de modificación
deben enviarse a `POST /api/chat` con el token en la cabecera
`X-Admin-Token`; la interfaz web no envía esa cabecera.

Entre las operaciones disponibles están `/recargar`, `/aprender texto` y
`/web URL`. `/web` solo acepta direcciones públicas HTTP o HTTPS. Protege el
acceso de red, los documentos y el token antes de habilitar estas operaciones.
No publiques el token ni guardes `.env` en el control de versiones.

## Solución de problemas

- **La página no abre:** ejecuta `docker compose ps` y revisa
  `docker compose logs -f unibot`.
- **El servicio sigue iniciando:** consulta `/health` y los registros. El
  primer inicio descarga modelos y puede tardar varios minutos.
- **Ollama no responde o descarga falla:** revisa
  `docker compose logs -f ollama` y confirma que Docker tiene Internet.
- **Responde que no hay información:** confirma que el dato esté escrito en
  un archivo de `conocimiento/`, reinicia UNIBOT y revisa las fuentes
  devueltas. Los ejemplos del clasificador no sustituyen esos documentos.
- **Cambiaste configuración:** ejecuta
  `docker compose up -d --force-recreate`.

## Limitaciones

- Las respuestas dependen de la calidad, cobertura y vigencia de los
  documentos locales.
- El endpoint de chat transmite fragmentos de texto durante la generación de
  Ollama; la ruta `POST /chat` sigue devolviendo una respuesta completa.
- El almacenamiento de sesiones es en memoria y no se comparte entre
  procesos ni sobrevive a un reinicio.
- Docker configura `CUDA_VISIBLE_DEVICES=-1`; TensorFlow usa CPU en esta
  configuración.
