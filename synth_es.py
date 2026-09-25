"""Spanish synthetic data generator for end-of-turn detection (the multilingual lane).

Same pipeline as synth.py: the same eleven classes, the same policy encodings
from POLICY.md, the same augmentation steps (truncation, context dropout,
ASR-style variant), the same gold-set collision guard. Pure code, seeded,
regenerates byte-identically. Writes data/train_es.jsonl and never touches
data/train.jsonl.

Register notes that matter:
- The target speaker is a Spanish-speaking carrier on a US freight lane, so
  the banks use Mexican-Spanish trucking register with real Spanglish: "el
  rate con", "la troca", "el dispatch", "el lumper", "el reefer". A purely
  textbook-Spanish dataset would miss how these calls actually sound.
- The hedge ownership ruling transfers cleanly: attributed claims ("el broker
  dijo que...", "me dijeron que...", "según ellos") mark the unsure stance and
  invite confirmation (speak); first-person claims with a decorative softener
  ("...o algo así", "...o algo por el estilo") are owned and mid-narrative
  (wait). Attribution markers exist in Spanish exactly as in English, which is
  why the ruling is trainable in both.
- The ASR variant also strips inverted punctuation. Telephony ASR does not
  reliably emit them, and stripping the opening question mark matters more in
  Spanish than stripping the final one does in English, because in Spanish it
  removes the question cue from the FRONT of the utterance.
- Agent context stays in Spanish (a bilingual agent answers in the caller's
  language), with the same "agent: ... caller: ..." input frame as English so
  one model serves both languages.
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

CITIES = ["Fontana", "Barstow", "Phoenix", "Denver", "Reno", "Ontario", "Stockton", "Laredo", "Memphis", "Dallas", "Atlanta", "Fresno", "El Paso", "Tucson", "Bakersfield", "McAllen"]
FACILITIES = ["la bodega", "el receiver", "el shipper", "el yard", "el dock cuatro", "el cross-dock", "la rampa dos"]
LOADS = ["cuatro siete dos", "ocho uno cinco", "dos dos nueve", "seis cuatro cero", "nueve uno tres", "tres ocho seis", "cinco cero cuatro"]
RATES = ["mil ochocientos cincuenta", "mil novecientos", "dos mil cien", "dos mil doscientos", "diecinueve cincuenta", "veintiuno cincuenta"]
TIMES = ["las ocho de la mañana", "el mediodía", "las tres y media", "las seis de la tarde", "mañana temprano", "el jueves en la mañana", "las cinco"]
EQUIP = ["el reefer", "la caja seca", "el flatbed", "el tráiler seis dos", "la troca"]
DOCS = ["el BOL", "la rate con", "el recibo del lumper", "el número de PO", "la confirmación"]
HIGHWAYS = ["la diez", "la quince", "la cuarenta", "la cinco", "la ocho"]
DIGITS = ["siete uno cinco", "cuatro uno cinco, cinco cinco", "tres tres cero, dos", "seis dos, ocho ocho", "nueve cero uno", "cinco cinco cuatro, siete"]
FILLERS = ["este", "pues", "o sea", "eh", "como que", "digo"]

AGENT_CTX = {
    "greet": ["Agent: Dispatch, ¿en qué le puedo ayudar?", "Agent: Habla dispatch, dígame."],
    "anything_else": ["Agent: ¿Algo más en que le pueda ayudar?", "Agent: ¿Alguna otra cosa?"],
    "rate": ["Agent: El rate es {rate}, todo incluido.", "Agent: Lo más que puedo ofrecer es {rate}."],
    "ask_mc": ["Agent: ¿Me da su número de MC?", "Agent: ¿Cuál es su MC?"],
    "ask_phone": ["Agent: ¿A qué número le regreso la llamada?", "Agent: ¿Cuál es su celular?"],
    "ask_where": ["Agent: ¿Dónde anda ahorita?", "Agent: ¿Cuál es su ubicación actual?"],
    "ask_appt": ["Agent: ¿Alcanza a llegar a la cita de {time}?", "Agent: ¿Le funciona el horario de {time}?"],
    "info": ["Agent: El dock cierra hoy a {time}.", "Agent: Su check call quedó para {time}.", "Agent: Le mando el BOL actualizado por correo."],
}


def fill(t: str, rng: random.Random) -> str:
    return t.format(
        city=rng.choice(CITIES), facility=rng.choice(FACILITIES), load=rng.choice(LOADS),
        rate=rng.choice(RATES), time=rng.choice(TIMES), equip=rng.choice(EQUIP),
        doc=rng.choice(DOCS), hwy=rng.choice(HIGHWAYS), digits=rng.choice(DIGITS),
        f1=rng.choice(FILLERS), f2=rng.choice(FILLERS),
    )


def ctx(kind: str | None, rng: random.Random) -> str:
    if kind is None:
        return ""
    return fill(rng.choice(AGENT_CTX[kind]), rng)


# Each entry: (class, label, context_kind or None, list of templates)
BANKS: list[tuple[str, str, str | None, list[str]]] = [
    # A: complete statements -> speak
    ("A", "speak", "greet", [
        "Llamo para confirmar la recogida de la carga {load} {time}.",
        "Acabo de dejar {equip} en el yard de {city}.",
        "Ya voy cargado saliendo de {city} ahorita.",
        "{equip} va jalando bien y voy a tiempo.",
        "Recogí la carga en {facility} esta mañana.",
        "Mi troca está vacía y ando buscando carga saliendo de {city}.",
    ]),
    ("A", "speak", None, [
        "Ya entregué en {facility} hace como una hora.",
        "La temperatura de {equip} va estable.",
        "Ya le mandé {doc} a su correo.",
        "El receiver ya firmó todo y ya voy de salida.",
        "Voy llegando a {city} como a {time}.",
    ]),
    # B: complete questions -> speak
    ("B", "speak", None, [
        "¿A qué hora cierra {facility}?",
        "¿Me puede mandar {doc} otra vez?",
        "¿Cubren la detención después de dos horas en el dock?",
        "¿Qué tan lejos queda {facility} de {city}?",
        "¿Necesito recibo del lumper para esta carga?",
        "¿Cuánto paga esa carga saliendo de {city}?",
        "¿A quién le llamo cuando llegue a {facility}?",
        "¿Puedo recoger antes si llego antes de {time}?",
    ]),
    # C: backchannel acks -> speak
    ("C", "speak", "info", [
        "Órale, está bien.", "Perfecto, gracias.", "Sale, entendido.", "Está bien.",
        "Sí, me funciona.", "Enterado.", "Va, gracias.", "Muy bien, gracias.",
    ]),
    # C: casual register acks and closers -> speak (matches the live-probe finding
    # in English: the model must see the short casual forms or it goes uncertain)
    ("C", "speak", "info", [
        "Simón.", "Sale, gracias.", "Va.", "Sí, sí, está bien.",
    ]),
    ("C", "speak", "anything_else", [
        "No, eso es todo, adiós.", "No, ya quedamos.", "No, ya con eso.",
        "Nomás eso, gracias, bye.", "Sí, gracias, hasta luego.", "No, eso cubre todo.",
        "No, ya, gracias.", "Eso es todo, que le vaya bien.",
    ]),
    # D: mid-clause cutoffs -> wait
    ("D", "wait", None, [
        "Ando como a veinte millas pero el tráfico en {hwy} está",
        "¿Le puede decir a {facility} que mi ETA ahora es",
        "El problema principal con esa ruta es el",
        "Después de entregar en {city} pensaba",
        "El broker de esa carga dijo que el",
        "Si el dock está lleno a lo mejor tengo que",
    ]),
    # E: disfluent trails -> wait
    ("E", "wait", None, [
        "Sí pues, {f1}, lo que pasa es que, {f2}",
        "Quería preguntarle de, {f1},",
        "Okay pues básicamente, {f1}, yo",
        "Bueno, {f1}, es como, {f2}",
        "Entonces, {f1}, sobre la entrega, {f2}",
    ]),
    # F: mid-data pauses -> wait
    ("F", "wait", "ask_mc", [
        "Sí, es {digits}", "Claro, MC {digits}", "Es {digits}, espere",
    ]),
    ("F", "wait", "ask_phone", [
        "Me puede marcar al {digits}", "Es {digits}", "Mi celular es {digits}",
    ]),
    # G: connector-final -> wait
    ("G", "wait", None, [
        "Puedo recoger {time}, pero",
        "El rate me funciona, aunque",
        "Normalmente agarraría esa carga, excepto que",
        "Podemos hacer el drop trailer, y",
        "Llego como a {time}, entonces",
    ]),
    # H: complete-then-maybe-more -> speak (policy: respond, barge-in covers the miss)
    ("H", "speak", "ask_where", ["Acabo de pasar {city}.", "Como a una hora de {city}.", "Aquí parado en {facility} ahorita."]),
    ("H", "speak", "ask_appt", ["Sí, sí llego.", "Me funciona.", "Sí, sin problema."]),
    ("H", "speak", "rate", ["Ese rate me funciona.", "Órale, la agarro.", "Trato hecho, apúntela."]),
    ("H", "speak", None, [
        "Yo le llamo a {facility} directamente.",
        "Mándeme los papeles a mi correo.",
        "Entregué la carga hace como una hora.",
        "Órale, voy para allá ahorita.",
    ]),
    # H exception: announced continuation -> wait
    ("H", "wait", "anything_else", [
        "Ah sí, una cosa más.",
        "Espere, hay algo más.",
        "Antes de que se me olvide, otra pregunta.",
        "Aguante, una última cosa.",
    ]),
    # I: handoff hedges -> speak (hedges are handoffs; settling hedges stay out of training)
    ("I", "speak", None, [
        "Eso es todo lo que ocupo, creo.",
        "Con eso queda todo, yo creo.",
        "Agarro la carga de {city}, supongo.",
        "Ya quedamos entonces, creo.",
    ]),
    # I hedge ruling: genuinely-unsure stance (reported speech, real doubt) invites confirmation -> speak
    ("I", "speak", None, [
        "El broker dijo que el lumper estaba cubierto, supuestamente.",
        "Me dijeron que el dock abre a {time}, según ellos.",
        "Dispatch dijo que el rate subió a {rate}, supuestamente.",
        "El shipper me dijo que es drop trailer, creo.",
    ]),
    # I hedge ruling: sure stance with a decorative softener, mid-narrative statement -> wait
    ("I", "wait", None, [
        "La detención ya quedó aprobada, o algo así",
        "La cita se movió a {time}, o algo por el estilo",
        "El rate salió en {rate}, o algo así",
        "El tráiler lo cambiaron en {facility}, o lo que haya sido",
    ]),
    ("I", "speak", None, [
        "Dijeron que el código del gate es el mismo de la otra vez, supuestamente.",
        "El receiver me dijo que puedo llegar antes, según él.",
        "Mi dispatcher dijo que usted ya tenía la rate con, creo.",
        "Eso sería todo de mi lado, creo.",
    ]),
    ("I", "wait", None, [
        "El código del gate lo cambiaron la semana pasada, o algo así",
        "El lumper salió en noventa dólares, o por ahí",
        "Salieron veintidós tarimas, o algo por el estilo",
    ]),
    # J: self-interrupt restarts -> wait
    ("J", "wait", None, [
        "Apúnteme en la- espere, aguante",
        "Ocupo el- no, momento",
        "Dígales que llego a- mmm, déjeme pensarlo",
        "La recogida es a- perdón, estoy leyendo la carga equivocada",
        "Póngame el check call a- bueno, de hecho",
        "Mi ETA es- ay, el GPS me acaba de cambiar la ruta",
    ]),
    # J exception: full retraction -> speak
    ("J", "speak", None, [
        "¿Me puede checar el- no, olvídelo.",
        "Mándeme el- no, déjelo, ya lo encontré.",
        "Apunte la- mejor no, yo lo hago en línea.",
    ]),
    # K: self-retrieval hold -> wait
    ("K", "wait", None, [
        "Aguante, déjeme buscar el número de la carga",
        "Un segundo, déjeme checar {doc}",
        "Déjeme buscar {doc} rapidito",
        "Deme un segundo para abrir la carga",
        "Eh, déjeme ver mi calendario",
        "Espere, déjeme leer el correo de confirmación",
    ]),
    # K: narrated external interruption -> speak (a brief courtesy ack is a response)
    ("K", "speak", None, [
        "Espere, me estoy orillando para poder leerlo",
        "Un segundo, el del dock me está haciendo señas",
        "Déjeme preguntarle a mi dispatcher y le regreso la llamada",
        "Me está entrando otra llamada, aguante",
        "Hay alguien en mi ventana, deme un minuto",
    ]),
    # General-assistant slices, same as the English generator carries
    ("A", "speak", None, [
        "Necesito mover mi cita del dentista para el jueves.",
        "El paquete por fin llegó esta mañana.",
        "Voy al gimnasio saliendo de esta llamada.",
    ]),
    ("B", "speak", None, [
        "¿Cómo va a estar el clima este fin de semana?",
        "¿Cuánto se hace al aeropuerto a las cinco?",
        "¿Llegó algo importante esta mañana?",
    ]),
    ("D", "wait", None, [
        "¿Me puede poner un recordatorio de lo del",
        "Le iba a preguntar del correo de",
    ]),
]


def asr_variant(text: str) -> str:
    """Lowercase and strip terminal AND inverted punctuation, as telephony ASR output arrives."""
    t = text.lower().strip()
    while t and t[-1] in ".?!":
        t = t[:-1].rstrip()
    t = t.replace("¿", "").replace("¡", "").strip()
    return t


def truncate(text: str, rng: random.Random) -> str | None:
    words = text.rstrip(".?!").split()
    if len(words) < 5:
        return None
    cut = rng.randint(max(2, int(len(words) * 0.4)), len(words) - 2)
    return " ".join(words[:cut])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/train_es.jsonl")
    ap.add_argument("--per-template", type=int, default=10, help="filled instances per template")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold", default="data/gold_set.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []

    # tpl is a stable template id (crc32 of the template string); augmented rows inherit it
    # so a grouped train/val split cannot leak slot fills of one template across the split.
    for cls, label, ctx_kind, templates in BANKS:
        for t in templates:
            tpl = f"es{zlib.crc32(t.encode())}"
            for _ in range(args.per_template):
                text = fill(t, rng)
                context = ctx(ctx_kind, rng)
                rows.append({"context": context, "text": text, "label": label, "cls": cls, "variant": "clean", "lang": "es", "tpl": tpl})

    # Truncation augmentation: complete speak utterances re-emitted cut short, labeled wait.
    speak_rows = [r for r in rows if r["label"] == "speak" and r["cls"] in ("A", "B", "H")]
    for r in speak_rows:
        cut = truncate(r["text"], rng)
        if cut:
            rows.append({"context": r["context"], "text": cut, "label": "wait", "cls": "T", "variant": "clean", "lang": "es", "tpl": r["tpl"]})

    # Context-dropout augmentation: every contexted sample also emitted bare.
    for r in list(rows):
        if r["context"]:
            rows.append({**r, "context": "", "variant": r["variant"] + "+noctx"})

    # ASR-style variant of everything.
    for r in list(rows):
        t = asr_variant(r["text"])
        if t != r["text"]:
            rows.append({**r, "text": t, "variant": "asr"})

    # Dedup and drop any exact-text collision with the frozen gold set.
    gold = json.load(open(args.gold))
    gold_texts = {s["text"].strip().lower() for s in gold["samples"]}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["context"], r["text"])
        if key in seen or r["text"].strip().lower() in gold_texts:
            continue
        seen.add(key)
        out.append(r)
    rng.shuffle(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_label = Counter(r["label"] for r in out)
    by_cls = Counter(r["cls"] for r in out)
    print(f"wrote {len(out)} samples to {args.out}")
    print("labels:", dict(by_label))
    print("classes:", dict(sorted(by_cls.items())))


if __name__ == "__main__":
    main()
