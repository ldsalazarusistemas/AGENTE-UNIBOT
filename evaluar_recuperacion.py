import json
from pathlib import Path

import app as bot

BASE = Path(__file__).resolve().parent
CASOS = BASE / "tests" / "evaluacion.json"


def evaluar() -> int:
    bot.cargar_indice()
    print("Sincronizando el índice con los documentos actuales...")
    print(bot.sincronizar())

    casos = json.loads(CASOS.read_text(encoding="utf-8"))
    aciertos = 0

    for caso in casos:
        resultados = bot.buscar_conocimiento(caso["pregunta"])
        fuentes = {resultado["fuente"] for resultado in resultados}
        acierto = caso["fuente_esperada"] in fuentes
        aciertos += int(acierto)
        estado = "OK" if acierto else "FALLO"
        print(
            f"{estado}: {caso['pregunta']} "
            f"(esperada: {caso['fuente_esperada']}; "
            f"recuperadas: {', '.join(sorted(fuentes)) or 'ninguna'})"
        )

    recall = aciertos / len(casos) if casos else 0.0
    print(f"Recall de fuente esperada: {aciertos}/{len(casos)} ({recall:.1%})")
    return 0 if aciertos == len(casos) else 1


if __name__ == "__main__":
    raise SystemExit(evaluar())
