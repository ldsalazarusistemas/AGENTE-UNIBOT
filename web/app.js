const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");
const statusText = document.getElementById("status-text");
const statusDot = document.querySelector(".status-dot");

const BIENVENIDA = messages.innerHTML;
const ALTURA_INPUT = 44;
const SVG = (trazos) =>
    `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${trazos}</svg>`;
const ICONO_ENLACE = SVG(
    '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/>' +
    '<path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>'
);
const ICONO_DOCUMENTO = SVG(
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>' +
    '<path d="M14 3v5h5"/>'
);
const ICONO_COPIAR = SVG(
    '<rect x="9" y="9" width="12" height="12" rx="2"/>' +
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
);

let sessionId = crearSesion();
let ocupado = false;

/* ---------- Utilidades ---------- */

function crearSesion() {
    if (window.crypto && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return "sesion-" + Date.now() + "-" +
        Math.random().toString(36).slice(2);
}

function el(etiqueta, clase, texto) {
    const nodo = document.createElement(etiqueta);
    if (clase) nodo.className = clase;
    if (texto !== undefined) nodo.textContent = texto;
    return nodo;
}

function bajar() {
    messages.scrollTop = messages.scrollHeight;
}

function actualizarBoton() {
    sendButton.disabled = ocupado || !input.value.trim();
}

function ajustarAltura() {
    input.style.height = ALTURA_INPUT + "px";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

/* ---------- Estado del servidor ---------- */

async function comprobarEstado() {
    try {
        const response = await fetch("/api/estado");
        if (!response.ok) throw new Error("API no disponible");

        const estado = await response.json();
        const listo = estado.listo === true;

        statusText.textContent = estado.mensaje || (
            listo ? "En línea" : "Preparando modelo…"
        );
        statusDot.classList.toggle("online", listo);
        statusDot.classList.remove("offline");
    } catch {
        statusText.textContent = "Servidor no disponible";
        statusDot.classList.remove("online");
        statusDot.classList.add("offline");
    }
}

/* ---------- Texto con formato básico (negrita, listas, enlaces) ---------- */

const URL_RE = /https?:\/\/[^\s<>"')\]]+/g;

function agregarEnlaces(padre, texto) {
    let ultimo = 0;
    for (const m of texto.matchAll(URL_RE)) {
        const url = m[0].replace(/[.,;:]+$/, "");
        padre.append(texto.slice(ultimo, m.index));
        const a = el("a", "", url);
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        padre.appendChild(a);
        ultimo = m.index + url.length;
    }
    padre.append(texto.slice(ultimo));
}

function agregarInline(padre, texto) {
    for (const parte of texto.split(/(\*\*[^*]+\*\*)/g)) {
        if (parte.length > 4 && parte.startsWith("**") && parte.endsWith("**")) {
            const negrita = el("strong");
            agregarEnlaces(negrita, parte.slice(2, -2));
            padre.appendChild(negrita);
        } else {
            agregarEnlaces(padre, parte);
        }
    }
}

function renderTexto(contenedor, texto) {
    contenedor.replaceChildren();
    let lista = null;

    for (const linea of texto.split("\n")) {
        const item = linea.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
        if (item) {
            if (!lista) {
                lista = el("ul");
                contenedor.appendChild(lista);
            }
            const li = el("li");
            agregarInline(li, item[1]);
            lista.appendChild(li);
            continue;
        }

        lista = null;
        if (!linea.trim()) continue;
        const p = el("p");
        agregarInline(p, linea);
        contenedor.appendChild(p);
    }
}

/* ---------- Mensajes ---------- */

function crearMensaje(tipo) {
    const bienvenida = document.getElementById("welcome");
    if (bienvenida) bienvenida.remove();

    const fila = el("div", `message ${tipo}`);
    if (tipo === "assistant") fila.appendChild(el("div", "avatar", "U"));

    const cuerpo = el("div", "message-body");
    const burbuja = el("div", "bubble");
    cuerpo.appendChild(burbuja);
    fila.appendChild(cuerpo);

    messages.appendChild(fila);
    bajar();
    return { fila, cuerpo, burbuja };
}

function mostrarEspera(burbuja, texto) {
    const puntos = el("span", "typing");
    puntos.append(el("i"), el("i"), el("i"));
    burbuja.classList.remove("error");
    burbuja.replaceChildren(puntos, el("span", "typing-text", texto));
}

function nombreLegible(nombre) {
    const base = String(nombre)
        .replace(/^.*[\\/]/, "")
        .replace(/\.[a-z0-9]+$/i, "")
        .replace(/[_-]+/g, " ")
        .trim();
    return base ? base.charAt(0).toUpperCase() + base.slice(1) : String(nombre);
}

function mostrarFuentes(cuerpo, fuentes) {
    if (!Array.isArray(fuentes) || fuentes.length === 0) return;

    const zona = el("div", "sources");
    zona.appendChild(el("span", "", "Fuentes:"));

    for (const fuente of fuentes) {
        const nombre = nombreLegible(fuente.nombre || fuente);
        let etiqueta;
        if (fuente.url) {
            etiqueta = el("a", "source");
            etiqueta.href = fuente.url;
            etiqueta.target = "_blank";
            etiqueta.rel = "noopener noreferrer";
            etiqueta.title = fuente.url;
            etiqueta.insertAdjacentHTML("beforeend", ICONO_ENLACE);
        } else {
            etiqueta = el("span", "source");
            etiqueta.insertAdjacentHTML("beforeend", ICONO_DOCUMENTO);
        }
        etiqueta.append(nombre);
        zona.appendChild(etiqueta);
    }

    cuerpo.appendChild(zona);
}

async function copiar(texto, boton) {
    try {
        await navigator.clipboard.writeText(texto);
    } catch {
        const auxiliar = el("textarea");
        auxiliar.value = texto;
        document.body.appendChild(auxiliar);
        auxiliar.select();
        document.execCommand("copy");
        auxiliar.remove();
    }
    const etiqueta = boton.querySelector("span");
    etiqueta.textContent = "Copiado";
    window.setTimeout(() => { etiqueta.textContent = "Copiar"; }, 1600);
}

function agregarAcciones(cuerpo, texto) {
    const zona = el("div", "actions");
    const boton = el("button", "action-btn");
    boton.type = "button";
    boton.insertAdjacentHTML("beforeend", ICONO_COPIAR);
    boton.appendChild(el("span", "", "Copiar"));
    boton.addEventListener("click", () => copiar(texto, boton));
    zona.appendChild(boton);
    cuerpo.appendChild(zona);
}

/* ---------- Envío y lectura del flujo NDJSON ---------- */

async function enviarPregunta(pregunta) {
    const texto = pregunta.trim();
    if (!texto || ocupado) return;

    ocupado = true;
    const usuario = crearMensaje("user");
    usuario.burbuja.textContent = texto;

    input.value = "";
    ajustarAltura();

    const respuesta = crearMensaje("assistant");
    const { burbuja, cuerpo } = respuesta;
    mostrarEspera(burbuja, "Consultando los documentos…");

    input.disabled = true;
    actualizarBoton();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message: texto })
        });

        if (!response.ok) throw new Error(`Error HTTP ${response.status}`);
        if (!response.body) {
            throw new Error("El servidor no devolvió un flujo de datos.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let contenido = "";
        let fuentes = [];
        let errorRecibido = false;

        const procesarLinea = (linea) => {
            if (!linea.trim()) return;

            let evento;
            try {
                evento = JSON.parse(linea);
            } catch {
                return;
            }

            if (evento.t === "fuentes") {
                fuentes = evento.d || [];
                if (fuentes.length) {
                    mostrarEspera(burbuja, "Generando respuesta…");
                }
            } else if (evento.t === "texto") {
                contenido += evento.d || "";
                renderTexto(burbuja, contenido);
            } else if (evento.t === "error") {
                errorRecibido = true;
                burbuja.classList.add("error");
                burbuja.textContent = evento.d || "Ocurrió un error.";
            }
            bajar();
        };

        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), {
                stream: !done
            });

            const lineas = buffer.split("\n");
            buffer = lineas.pop() || "";
            lineas.forEach(procesarLinea);

            if (done) break;
        }
        if (buffer.trim()) procesarLinea(buffer);

        if (!contenido && !errorRecibido) {
            burbuja.classList.add("error");
            burbuja.textContent = "El servidor no devolvió una respuesta.";
        }

        if (!errorRecibido) mostrarFuentes(cuerpo, fuentes);
        if (contenido && !errorRecibido) agregarAcciones(cuerpo, contenido);
    } catch (error) {
        burbuja.classList.add("error");
        burbuja.textContent =
            "No pude comunicarme con UNIBOT. Revisa que el servidor esté " +
            "encendido y vuelve a intentarlo.";
        console.error(error);
    } finally {
        ocupado = false;
        input.disabled = false;
        actualizarBoton();
        input.focus();
        bajar();
    }
}

function nuevaConversacion() {
    sessionId = crearSesion();
    messages.innerHTML = BIENVENIDA;
    input.focus();
}

/* ---------- Eventos ---------- */

form.addEventListener("submit", (event) => {
    event.preventDefault();
    enviarPregunta(input.value);
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!sendButton.disabled) form.requestSubmit();
    }
});

input.addEventListener("input", () => {
    ajustarAltura();
    actualizarBoton();
});

document.addEventListener("click", (event) => {
    const atajo = event.target.closest("[data-q]");
    if (atajo) {
        enviarPregunta(atajo.dataset.q);
        return;
    }
    if (event.target.closest("[data-nuevo]")) nuevaConversacion();
});

comprobarEstado();
window.setInterval(comprobarEstado, 15000);
input.focus();