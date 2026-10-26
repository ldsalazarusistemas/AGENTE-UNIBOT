import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

_test_directory = tempfile.TemporaryDirectory()
os.environ["DOCS_DIR"] = os.path.join(_test_directory.name, "docs")
os.environ["DATA_DIR"] = os.path.join(_test_directory.name, "data")

embeddings_stub = types.ModuleType("embeddings_tf")
embeddings_stub.embed = lambda _texts: np.zeros((1, 512), dtype=np.float32)
generator_stub = types.ModuleType("generador")
generator_stub.generar = lambda *_args, **_kwargs: "Respuesta de prueba."
generator_stub.generar_stream = lambda *_args, **_kwargs: iter(
    ["Respuesta de prueba."]
)
bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = object
pypdf_stub = types.ModuleType("pypdf")
pypdf_stub.PdfReader = object

sys.modules["embeddings_tf"] = embeddings_stub
sys.modules["generador"] = generator_stub
sys.modules["bs4"] = bs4_stub
sys.modules["pypdf"] = pypdf_stub

import app as bot

del sys.modules["generador"]
import generador as generador_real


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        bot.ENTRADAS = [
            {
                "fuente": "admisiones.md",
                "texto": "Convocatoria oficial: https://unintep.edu.co/portal/admisiones/",
            },
            {"fuente": "general.md", "texto": "Información general."},
            {"fuente": "costos.md", "texto": "Tarifas institucionales."},
        ]
        bot.VECTORES = np.array(
            [[1.0, 0.0], [0.4, 0.9165], [0.1, 0.9949]],
            dtype=np.float32,
        )

    def test_trocear_keeps_markdown_heading_with_its_content(self):
        text = (
            "Introducción breve que llena parte del primer fragmento.\n\n"
            "## Misión de UNINTEP\n\n"
            "Somos una institución comprometida con la excelencia académica."
        )

        chunks = bot.trocear(text, max_chars=80, solape=10)
        mission_chunk = next(
            chunk for chunk in chunks if "Somos una institución" in chunk
        )

        self.assertIn("## Misión de UNINTEP", mission_chunk)

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_discards_results_below_similarity_threshold(self, _embed):
        result = bot.buscar_conocimiento("admisiones", k=3, umbral=0.5)

        self.assertEqual([item["fuente"] for item in result], ["admisiones.md"])
        self.assertGreaterEqual(result[0]["similitud"], 0.5)

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_uses_documented_default_similarity_threshold(self, _embed):
        bot.ENTRADAS = [
            {"fuente": "general.md", "texto": "Información institucional."},
            {"fuente": "costos.md", "texto": "Tarifas institucionales."},
        ]
        bot.VECTORES = np.array(
            [[0.25, 0.9682], [0.1, 0.995]], dtype=np.float32
        )

        with patch.object(bot, "MIN_SIMILITUD", 0.20):
            result = bot.buscar_conocimiento("xyz")

        self.assertEqual([item["fuente"] for item in result], ["general.md"])

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_combines_semantic_and_keyword_matches_with_source_diversity(
        self,
        _embed,
    ):
        bot.ENTRADAS = [
            {
                "fuente": "preguntasfrecuentes.md",
                "texto": "¿Cuál es el teléfono general de UNINTEP?",
            },
            {
                "fuente": "general.md",
                "texto": "Teléfonos generales publicados en UNINTEP.",
            },
            {
                "fuente": "preguntasfrecuentes.md",
                "texto": "Otros datos institucionales.",
            },
            {
                "fuente": "directorio.md",
                "texto": "Contacto general. Teléfono: +57 300 000 0000.",
            },
        ]
        bot.VECTORES = np.array(
            [[0.33, 0.0], [0.26, 0.0], [0.243, 0.0], [0.165, 0.0]],
            dtype=np.float32,
        )

        result = bot.buscar_conocimiento(
            "¿Cuál es el teléfono general de UNINTEP?",
            k=3,
            umbral=0.2,
        )

        fuentes = [item["fuente"] for item in result]
        self.assertEqual(fuentes[0], "preguntasfrecuentes.md")
        self.assertIn("general.md", fuentes)
        self.assertIn("directorio.md", fuentes)
        self.assertEqual(len(fuentes), len(set(fuentes)))

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_does_not_overweight_generic_incidental_terms(self, _embed):
        bot.ENTRADAS = [
            {"fuente": "tramites.md", "texto": "Certificados de estudio."},
            {
                "fuente": "biblioteca.md",
                "texto": "Préstamo de libros y recursos bibliográficos.",
            },
        ]
        bot.VECTORES = np.array(
            [[0.208, 0.978], [0.273, 0.962]], dtype=np.float32
        )

        result = bot.buscar_conocimiento(
            "¿Puedo reservar una sala de estudio?",
            k=2,
            umbral=0.20,
        )

        self.assertEqual(result[0]["fuente"], "biblioteca.md")

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_prefers_source_diversity_then_returns_other_fragments(
        self,
        _embed,
    ):
        bot.ENTRADAS = [
            {"fuente": "biblioteca.md", "texto": "Préstamo de libros."},
            {"fuente": "directorio.md", "texto": "Teléfono institucional."},
            {"fuente": "biblioteca.md", "texto": "Horario de biblioteca."},
        ]
        bot.VECTORES = np.array(
            [[0.9, 0.0], [0.7, 0.0], [0.8, 0.0]], dtype=np.float32
        )

        result = bot.buscar_conocimiento(
            "biblioteca",
            k=3,
            umbral=0.2,
        )

        self.assertEqual(
            [item["fuente"] for item in result],
            ["biblioteca.md", "directorio.md", "biblioteca.md"],
        )
        self.assertEqual(len({item["texto"] for item in result}), 3)

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_search_prioritizes_all_relevant_chunks_from_intent_source(
        self,
        _embed,
    ):
        bot.ENTRADAS = [
            {"fuente": "preguntasfrecuentes.md", "texto": "Oferta académica."},
            {"fuente": "programas.md", "texto": "Programas de administración."},
            {"fuente": "programas.md", "texto": "Programas de sistemas."},
            {"fuente": "programas.md", "texto": "Programas de posgrado."},
        ]
        bot.VECTORES = np.array(
            [[0.95, 0.0], [0.70, 0.0], [0.65, 0.0], [0.60, 0.0]],
            dtype=np.float32,
        )

        result = bot.buscar_conocimiento(
            "¿Qué programas ofrece UNINTEP?",
            k=4,
            umbral=0.2,
            boost={"programas.md": 0.08},
        )

        self.assertEqual(
            [item["fuente"] for item in result],
            [
                "programas.md",
                "programas.md",
                "programas.md",
                "preguntasfrecuentes.md",
            ],
        )

    @patch.object(bot, "embed", return_value=np.array([[1.0, 0.0]]))
    def test_rare_exact_query_term_promotes_specific_evidence(self, _embed):
        bot.ENTRADAS = [
            {
                "fuente": "preguntasfrecuentes.md",
                "texto": "¿Qué carreras ofrece UNINTEP? Consulta el catálogo oficial.",
            },
            {
                "fuente": "programas.md",
                "texto": (
                    "Sistemas y Electricidad. Profesional universitario: "
                    "Ingeniería Informática."
                ),
            },
        ]
        bot.VECTORES = np.array([[0.52, 0.0], [0.50, 0.0]], dtype=np.float32)

        result = bot.buscar_conocimiento(
            "¿Qué carreras de ingeniería ofrece UNINTEP?",
            k=2,
            umbral=0.2,
        )

        self.assertEqual(result[0]["fuente"], "programas.md")

    def test_search_terms_match_common_singular_and_plural_forms(self):
        self.assertIn("libro", bot._terminos("libros"))

    @patch.object(bot, "buscar_conocimiento")
    def test_context_returns_linked_deduplicated_sources(self, search):
        search.return_value = [
            {
                "fuente": "admisiones.md",
                "texto": (
                    "Admisiones. "
                    "https://unintep.edu.co/portal/admisiones/"
                ),
            },
            {
                "fuente": "admisiones.md",
                "texto": "Otro fragmento de la misma fuente.",
            },
        ]

        _context, sources = bot.construir_contexto("¿Cómo me inscribo?", [])

        self.assertEqual(
            sources,
            [{
                "nombre": "admisiones.md",
                "url": "https://unintep.edu.co/portal/admisiones/",
            }],
        )

    @patch.object(
        bot,
        "buscar_conocimiento",
        return_value=[{
            "fuente": "preguntasfrecuentes.md",
            "texto": (
                "Respuesta. https://example.org/admisiones "
                "https://example.org/programas"
            ),
        }],
    )
    def test_context_omits_ambiguous_source_url(self, _search):
        _context, sources = bot.construir_contexto("Pregunta", [])

        self.assertEqual(
            sources,
            [{"nombre": "preguntasfrecuentes.md", "url": ""}],
        )

    @patch.object(bot, "construir_contexto", return_value=("", []))
    @patch.object(bot, "generar")
    def test_no_evidence_skips_generator_and_returns_sources(
        self,
        generate,
        _context,
    ):
        history = [{"role": "system", "content": "Instrucciones."}]

        response, sources = bot.responder_con_fuentes(
            history,
            "¿Cuál es la información?",
        )

        self.assertEqual(response, bot.RESPUESTA_SIN_EVIDENCIA)
        self.assertEqual(sources, [])
        generate.assert_not_called()
        self.assertEqual(history[-1]["content"], response)

    @patch.object(bot, "cargar_instrucciones", return_value="Instrucciones.")
    @patch.object(
        bot,
        "construir_contexto",
        return_value=(
            "Contexto.",
            [{"nombre": "general.md", "url": ""}],
        ),
    )
    @patch.object(bot, "generar_stream", return_value=iter(["Res", "puesta."]))
    def test_streaming_response_passes_history_and_updates_it(
        self,
        generate,
        _context,
        _instructions,
    ):
        history = [
            {"role": "system", "content": "Instrucciones anteriores."},
            {"role": "user", "content": "¿Qué programas hay?"},
            {"role": "assistant", "content": "Consulta el catálogo."},
        ]
        history_seen_by_generator = []

        def generate_response(*_args, **kwargs):
            history_seen_by_generator.extend(kwargs["historial"])
            return iter(["Res", "puesta."])

        generate.side_effect = generate_response

        events = list(bot.responder_stream(history, "¿Y cuánto duran?"))
        fragmentos = [
            event["d"] for event in events if event["t"] == "texto"
        ]

        self.assertEqual(
            events[0],
            {
                "t": "fuentes",
                "d": [{"nombre": "general.md", "url": ""}],
            },
        )
        self.assertEqual(fragmentos, ["Res", "puesta."])
        self.assertEqual(history[-2]["content"], "¿Y cuánto duran?")
        self.assertEqual(history[-1]["content"], "Respuesta.")
        self.assertEqual(history_seen_by_generator, history[:-2])

    @patch.object(bot, "construir_contexto", return_value=("", []))
    @patch.object(bot, "generar")
    def test_stream_without_evidence_does_not_call_generator(
        self,
        generate,
        _context,
    ):
        history = [{"role": "system", "content": "Instrucciones."}]

        events = list(bot.responder_stream(history, "Pregunta sin evidencia"))

        self.assertEqual(events[0], {"t": "fuentes", "d": []})
        self.assertEqual(
            "".join(event["d"] for event in events if event["t"] == "texto"),
            bot.RESPUESTA_SIN_EVIDENCIA,
        )
        generate.assert_not_called()

    def test_prompt_contains_recent_conversation_turns(self):
        prompt = generador_real.construir_prompt(
            "¿Y cuánto duran?",
            "Fragmento de evidencia.",
            "Instrucciones.",
            historial=[
                {"role": "system", "content": "No debe repetirse."},
                {"role": "user", "content": "¿Qué programas hay?"},
                {"role": "assistant", "content": "Consulta el catálogo."},
            ],
        )

        self.assertIn("Usuario: ¿Qué programas hay?", prompt)
        self.assertIn("Asistente: Consulta el catálogo.", prompt)
        self.assertIn("PREGUNTA:\n¿Y cuánto duran?", prompt)
        self.assertIn("enumera todos los nombres", prompt)
        self.assertIn("Incluye también los programas técnicos y tecnológicos", prompt)
        self.assertIn("no lo asignes a otra", prompt)
        self.assertNotIn("No debe repetirse.", prompt)

    def test_generator_sends_prompt_and_token_limit_to_ollama(self):
        tags = types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [{"name": "qwen2.5:3b"}]},
        )
        generated = types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "Respuesta breve."},
        )
        with (
            patch.object(generador_real, "_modelo_disponible", False),
            patch.object(generador_real.requests, "get", return_value=tags),
            patch.object(generador_real.requests, "post", return_value=generated) as post,
            patch.object(generador_real, "MAX_TOKENS", 32),
        ):
            respuesta = generador_real.generar(
                "Pregunta",
                "Contexto",
                "Instrucciones",
            )

        self.assertEqual(respuesta, "Respuesta breve.")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(payload["options"], {"num_predict": 32})
        self.assertFalse(payload["stream"])
        self.assertIn("Contexto", payload["prompt"])

    def test_generator_stream_emits_ollama_chunks_as_received(self):
        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self, chunk_size=512):
                self.chunk_size = chunk_size
                yield b'{"response":"Hola","done":false}'
                yield b'{"response":" mundo","done":false}'
                yield b'{"response":"","done":true}'

        stream = FakeStream()
        with (
            patch.object(generador_real, "_modelo_disponible", True),
            patch.object(
                generador_real.requests,
                "post",
                return_value=stream,
            ) as post,
        ):
            fragmentos = list(
                generador_real.generar_stream(
                    "Pregunta",
                    "Contexto",
                    "Instrucciones",
                )
            )

        self.assertEqual(fragmentos, ["Hola", " mundo"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(stream.chunk_size, 1)

    def test_generator_downloads_model_when_missing(self):
        tags = types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": []},
        )

        class FakeDownload:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield b'{"status":"success"}'

        with (
            patch.object(generador_real, "_modelo_disponible", False),
            patch.object(generador_real.requests, "get", return_value=tags),
            patch.object(
                generador_real.requests,
                "post",
                return_value=FakeDownload(),
            ) as post,
        ):
            generador_real.cargar_modelo()
            self.assertTrue(generador_real._modelo_disponible)

        self.assertEqual(
            post.call_args.kwargs["json"],
            {"name": "qwen2.5:3b", "stream": True},
        )

    def test_generator_reports_ollama_connection_failure(self):
        with (
            patch.object(generador_real, "_modelo_disponible", False),
            patch.object(
                generador_real.requests,
                "get",
                side_effect=generador_real.requests.ConnectionError("offline"),
            ),
            self.assertRaises(generador_real.ErrorGenerador),
        ):
            generador_real.cargar_modelo()


if __name__ == "__main__":
    unittest.main()
