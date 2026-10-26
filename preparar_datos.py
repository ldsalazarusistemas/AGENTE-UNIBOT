import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

EJEMPLOS = [
    # programas
    ("¿Qué programas ofrece la universidad?", "programas"),
    ("Quiero conocer la oferta académica", "programas"),
    ("¿Qué carreras puedo estudiar?", "programas"),
    ("¿Cuánto dura la carrera de ingeniería?", "programas"),
    ("Qué tecnologías y especializaciones hay", "programas"),
    ("Información del plan de estudios", "programas"),
    # admisiones
    ("¿Cómo me inscribo?", "admisiones"),
    ("¿Cuáles son los requisitos de admisión?", "admisiones"),
    ("¿Cuándo comienzan las inscripciones?", "admisiones"),
    ("Quiero ingresar a la universidad, qué debo hacer", "admisiones"),
    ("Documentos para matricularme como estudiante nuevo", "admisiones"),
    ("Cómo es el proceso de admisión", "admisiones"),
    # biblioteca
    ("¿Dónde está la biblioteca?", "biblioteca"),
    ("¿Qué servicios presta la biblioteca?", "biblioteca"),
    ("¿Cuál es el horario de la biblioteca?", "biblioteca"),
    ("Cómo presto un libro", "biblioteca"),
    ("Bases de datos y recursos bibliográficos", "biblioteca"),
    ("Puedo reservar una sala de estudio", "biblioteca"),
    # costos
    ("¿Cuánto cuesta la matrícula?", "costos"),
    ("Valor del semestre", "costos"),
    ("¿Hay becas o descuentos?", "costos"),
    ("Cómo puedo pagar la matrícula", "costos"),
    ("Tarifas y derechos pecuniarios", "costos"),
    ("Financiación y créditos educativos", "costos"),
    # calendario
    ("¿Cuándo empiezan las clases?", "calendario"),
    ("Fechas del calendario académico", "calendario"),
    ("¿Cuándo son los exámenes finales?", "calendario"),
    ("Qué día termina el semestre", "calendario"),
    ("Fechas de vacaciones y festivos", "calendario"),
    ("Cuándo es la entrega de notas", "calendario"),
    # bienestar
    ("¿Qué es bienestar universitario?", "bienestar"),
    ("Hay apoyo psicológico para estudiantes", "bienestar"),
    ("Actividades deportivas y culturales", "bienestar"),
    ("Servicio de salud y orientación", "bienestar"),
    ("Cómo participo en grupos de la universidad", "bienestar"),
    ("Apoyos para estudiantes en dificultades", "bienestar"),
    # prácticas y grados
    ("¿Cómo hago mis prácticas profesionales?", "practicas"),
    ("Requisitos para graduarme", "practicas"),
    ("Cómo es el proceso de grado", "practicas"),
    ("Opciones de trabajo de grado", "practicas"),
    ("Cuándo son las ceremonias de grado", "practicas"),
    ("Convenios para realizar pasantías", "practicas"),
    # trámites
    ("¿Cómo solicito un certificado de estudio?", "tramites"),
    ("Cómo pido una constancia de notas", "tramites"),
    ("Procedimiento para cancelar una materia", "tramites"),
    ("Cómo hago un reintegro o aplazamiento", "tramites"),
    ("Solicitar homologación de materias", "tramites"),
    ("Trámites académicos y administrativos", "tramites"),
    # contacto
    ("¿Cuál es el teléfono de la universidad?", "contacto"),
    ("Correo de contacto de admisiones", "contacto"),
    ("Dónde queda la sede y cómo llego", "contacto"),
    ("Con quién puedo hablar sobre mi caso", "contacto"),
    ("Directorio de dependencias", "contacto"),
    ("Redes sociales y página oficial", "contacto"),
    # saludo
    ("Hola, buenos días", "saludo"),
    ("Buenas tardes", "saludo"),
    ("Hola UNIBOT", "saludo"),
    ("Buenas noches", "saludo"),
    ("Hola", "saludo"),
    ("Qué tal, buen día", "saludo"),
    # agradecimiento
    ("Muchas gracias", "agradecimiento"),
    ("Te agradezco la información", "agradecimiento"),
    ("Gracias por ayudarme", "agradecimiento"),
    ("Mil gracias", "agradecimiento"),
    ("Gracias, eso era todo", "agradecimiento"),
    ("Perfecto, muchas gracias", "agradecimiento"),
    # general
    ("¿Qué es UNINTEP?", "general"),
    ("Dame información de la institución", "general"),
    ("¿Dónde puedo consultar información oficial?", "general"),
    ("Historia y misión de la universidad", "general"),
    ("Acreditación y reconocimiento de la institución", "general"),
    ("Cuéntame sobre la universidad", "general"),
]


def guardar_ejemplos():
    ruta = DATA / "intenciones.json"
    ruta.write_text(
        json.dumps(
            [{"texto": t, "intencion": i} for t, i in EJEMPLOS],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Se guardaron {len(EJEMPLOS)} ejemplos en {ruta}")


if __name__ == "__main__":
    guardar_ejemplos()