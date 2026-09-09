"""Calibración de umbral y margen con vídeo de laboratorio: mide el pipeline real, no un doble.

Verdad de referencia sin anotación manual:
- Positivo: dos recortes del mismo track separados por al menos `--gap` frames.
- Negativo: dos recortes de tracks distintos en el mismo frame con cajas sin solape (IoU < 0.1), para no
  contar como negativo la misma instancia detectada dos veces.
Los tracks cortos se descartan y se exporta un montaje por track para inspección visual de la verdad usada.
"""
import argparse
import itertools
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from odin_vision.config import Settings
from odin_vision.tracking import Tracker, iou
from odin_vision.vision import ModelVision, crop, quality


def frames(path, stride, limit, width):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    index = kept = 0
    try:
        while capture.isOpened() and kept < limit:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if (index-1) % stride:
                continue
            height, original = frame.shape[:2]
            factor = min(1.0, width/original)
            if factor < 1:
                frame = cv2.resize(frame, (round(original*factor), round(height*factor)))
            kept += 1
            yield kept, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def collect(vision, path, categories, stride, limit, width, min_confidence):
    """Devuelve {track_id: [(frame, embedding, recorte)]} usando el tracker del producto."""
    tracker = Tracker(max_age=3)
    observations = defaultdict(list)
    for number, image in frames(path, stride, limit, width):
        detections = [d for d in vision.detect(image, categories) if d.confidence >= min_confidence]
        for track in tracker.update(detections, number):
            if track.last_seen != number:
                continue
            patch = crop(image, track.box)
            if not quality(patch):
                continue
            key = f"{path.stem}#{track.id}"
            observations[key].append((number, vision.embed(patch), patch, track.box))
    return observations


def montage(observations, destination, per_track=5, size=96):
    from PIL import Image

    tracks = sorted(observations.items())
    sheet = Image.new("RGB", (per_track*size, len(tracks)*size), "black")
    for row, (_, items) in enumerate(tracks):
        picks = np.linspace(0, len(items)-1, per_track).astype(int)
        for column, index in enumerate(picks):
            sheet.paste(items[index][2].resize((size, size)), (column*size, row*size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return [name for name, _ in tracks]


def pairs(observations, gap, gallery, cap, seed):
    """Positivos: mismo track con separación temporal. Negativos: tracks coincidentes en un frame."""
    rng = random.Random(seed)
    positive, negative = [], []
    for items in observations.values():
        for (frame_a, *_), (frame_b, probe, *_) in itertools.combinations(items, 2):
            if frame_b - frame_a < gap:
                continue
            views = [vector for number, vector, *_ in items if number <= frame_a][-gallery:]
            positive.append(max(float(probe @ view) for view in views))
    by_frame = defaultdict(list)
    for name, items in observations.items():
        for number, vector, _, box in items:
            by_frame[number].append((name, vector, box))
    for entries in by_frame.values():
        for (name_a, a, box_a), (name_b, b, box_b) in itertools.combinations(entries, 2):
            # Cajas solapadas en el mismo frame suelen ser la misma instancia detectada dos veces.
            if name_a != name_b and iou(box_a, box_b) < 0.1:
                negative.append(float(a @ b))
    rng.shuffle(positive)
    rng.shuffle(negative)
    return positive[:cap], negative[:cap]


def rivals(observations):
    """Tracks que coinciden en algún frame: identidades distintas garantizadas, sin suponer nada del encoder."""
    present = defaultdict(set)
    for name, items in observations.items():
        for number, *_ in items:
            present[number].add(name)
    together = defaultdict(set)
    for names in present.values():
        for name in names:
            together[name] |= names - {name}
    return together


def identification(observations, gap, gallery, only_rivals=True):
    """Identificación en conjunto cerrado: cada track es una etiqueta con su galería, como en la memoria.

    Devuelve, por sonda, el parecido con su propia identidad y con la mejor identidad ajena. La diferencia entre
    ambas es lo que el guardia de ambigüedad (`margin`) exige superar. Con `only_rivals`, las ajenas se limitan
    a los tracks que aparecen simultáneamente con la sonda, que son necesariamente otras instancias; sin esa
    restricción, un mismo peatón dividido en dos tracks cuenta como confusión y el resultado sale pesimista.
    """
    galleries = {name: [vector for _, vector, *_ in items[:gallery]] for name, items in observations.items()}
    competitors = rivals(observations)
    rows = []
    for name, items in observations.items():
        others = competitors[name] if only_rivals else set(galleries) - {name}
        if not others:
            continue
        for number, probe, *_ in items:
            if number - items[0][0] < gap:
                continue
            own = max(float(probe @ view) for view in galleries[name])
            best_other, best_other_score = max(
                ((other, max(float(probe @ view) for view in galleries[other])) for other in others),
                key=lambda item: item[1])
            rows.append({"own": own, "best_other": best_other_score, "correct_top1": own > best_other_score,
                         "confused_with": best_other})
    return rows


def margin_curve(rows, threshold):
    """Cuántas identificaciones sobreviven al margen y cuántas confusiones se cuelan, para cada margen."""
    points = []
    for margin in np.round(np.arange(0.0, 0.31, 0.01), 2):
        accepted_correct = sum(r["correct_top1"] and r["own"] >= threshold and r["own"]-r["best_other"] >= margin
                               for r in rows)
        accepted_wrong = sum((not r["correct_top1"]) and r["best_other"] >= threshold and
                             r["best_other"]-r["own"] >= margin for r in rows)
        points.append({"margin": float(margin), "accepted_correct": int(accepted_correct),
                       "accepted_wrong": int(accepted_wrong),
                       "recall": float(accepted_correct/len(rows))})
    return points


def curve(positive, negative):
    points = []
    for threshold in np.round(np.arange(0.20, 0.98, 0.01), 2):
        true_positive = sum(score >= threshold for score in positive)
        false_positive = sum(score >= threshold for score in negative)
        recall = true_positive/len(positive)
        precision = true_positive/(true_positive+false_positive) if true_positive+false_positive else 1.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        points.append({"threshold": float(threshold), "recall": recall, "precision": precision,
                       "false_positive_rate": false_positive/len(negative), "f1": f1})
    return points


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(array, [1, 5, 50, 95, 99])
    return {"count": int(array.size), "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "p1": float(percentiles[0]), "p5": float(percentiles[1]),
            "median": float(percentiles[2]), "p95": float(percentiles[3]), "p99": float(percentiles[4])}


def main():
    parser = argparse.ArgumentParser(description="Calibra umbral y margen con material propio de laboratorio")
    parser.add_argument("--video", action="append", required=True, metavar="RUTA:categoria,categoria",
                        help="Vídeo local y vocabulario del detector; repetible")
    parser.add_argument("--stride", type=int, default=5, help="Analizar uno de cada N frames")
    parser.add_argument("--max-frames", type=int, default=60, help="Frames analizados por vídeo")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-track", type=int, default=4, help="Descartar tracks con menos observaciones")
    parser.add_argument("--gap", type=int, default=3, help="Separación mínima entre positivos, en frames analizados")
    parser.add_argument("--gallery", type=int, default=4, help="Vistas registradas por instancia, como en la memoria")
    parser.add_argument("--pairs", type=int, default=4000, help="Máximo de pares por clase")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("data/lab/calibration.json"))
    args = parser.parse_args()

    vision = ModelVision(replace(Settings.from_env(), backend="models"))
    observations = {}
    for entry in args.video:
        path, _, vocabulary = entry.rpartition(":")
        categories = [c.strip() for c in vocabulary.split(",") if c.strip()]
        found = collect(vision, Path(path), categories, args.stride, args.max_frames, args.width,
                        args.min_confidence)
        observations.update({k: v for k, v in found.items() if len(v) >= args.min_track})
        print(f"{path}: {len(found)} tracks, {sum(len(v) for v in found.values())} observaciones")

    if len(observations) < 2:
        raise SystemExit("Material insuficiente: se necesitan al menos dos tracks utilizables")
    sheet = args.out.with_suffix(".tracks.png")
    order = montage(observations, sheet)
    positive, negative = pairs(observations, args.gap, args.gallery, args.pairs, args.seed)
    if not positive or not negative:
        raise SystemExit("No se generaron pares suficientes; ajusta --gap, --stride o --min-track")

    points = curve(positive, negative)
    rows = identification(observations, args.gap, args.gallery)
    open_rows = identification(observations, args.gap, args.gallery, only_rivals=False)
    best_f1 = max(points, key=lambda p: p["f1"])
    clean = [p for p in points if p["false_positive_rate"] == 0]
    strict = min(clean, key=lambda p: p["threshold"]) if clean else None
    separation = min(positive) - max(negative)
    report = {
        "tracks": len(observations), "track_order": order, "montage": str(sheet),
        "positives": summarize(positive), "negatives": summarize(negative),
        "separation_min_positive_minus_max_negative": separation,
        "recommended": {
            "best_f1": best_f1,
            "lowest_threshold_without_false_positives": strict,
            "suggested_margin": round(max(0.02, min(0.15, (max(negative)-np.mean(negative))/2)), 3),
        },
        "identification": {
            "probes": len(rows),
            "top1_accuracy": float(sum(r["correct_top1"] for r in rows)/len(rows)) if rows else None,
            "own_scores": summarize([r["own"] for r in rows]) if rows else None,
            "best_other_scores": summarize([r["best_other"] for r in rows]) if rows else None,
            "gap_own_minus_other": summarize([r["own"]-r["best_other"] for r in rows]) if rows else None,
            "margin_curve": margin_curve(rows, best_f1["threshold"]) if rows else None,
            "all_tracks_as_rivals": {
                "probes": len(open_rows),
                "top1_accuracy": float(sum(r["correct_top1"] for r in open_rows)/len(open_rows))
                if open_rows else None,
                "note": "pesimista: un mismo peatón dividido en varios tracks cuenta como confusión",
            },
        },
        "curve": points,
        "settings": {k: getattr(args, k) for k in
                     ("stride", "max_frames", "width", "min_confidence", "min_track", "gap", "gallery", "seed")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("tracks", "positives", "negatives", "separation_min_positive_minus_max_negative",
                       "recommended")}, indent=2))
    identity = report["identification"]
    print(f"identificación: top-1 {identity['top1_accuracy']:.3f} sobre {identity['probes']} sondas; "
          f"diferencia con la mejor identidad ajena: {json.dumps(identity['gap_own_minus_other'])}")
    for point in identity["margin_curve"]:
        if point["margin"] in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2):
            print(f"  margen {point['margin']:.2f}: aciertos aceptados {point['accepted_correct']}"
                  f" ({point['recall']:.3f}), confusiones aceptadas {point['accepted_wrong']}")
    print("Montaje de verdad de referencia:", sheet)


if __name__ == "__main__":
    main()
