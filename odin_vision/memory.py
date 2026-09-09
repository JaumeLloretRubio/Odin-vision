"""Memoria vectorial y episódica transaccional, acotada y persistente en SQLite."""
import json
import sqlite3
import time

import numpy as np

from .vision import normalized


class Memory:
    def __init__(self, settings, fingerprint):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.db = sqlite3.connect(settings.data_dir / "memory.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, category TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY, label_id INTEGER REFERENCES labels(id) ON DELETE CASCADE,
                vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                session TEXT NOT NULL, track INTEGER NOT NULL, category TEXT NOT NULL,
                label TEXT, confidence REAL, location TEXT, kind TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS event_label_time ON events(label, timestamp);
        """)
        old = self.db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()
        if old and old[0] != fingerprint:
            self.db.close()
            raise ValueError("El encoder ha cambiado: use otro ODIN_DATA_DIR o exporte su memoria")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('encoder', ?)", (fingerprint,))
        self.db.commit()
        self.revision = 0

    def close(self):
        self.db.close()

    def learn(self, name, category, vectors):
        vectors = [normalized(v) for v in vectors]
        if not vectors:
            raise ValueError("No hay observaciones con calidad suficiente")
        dim = self.db.execute("SELECT value FROM metadata WHERE key='dimension'").fetchone()
        expected = int(dim[0]) if dim else len(vectors[0])
        if any(len(v) != expected for v in vectors):
            raise ValueError("Dimensión de embedding incompatible")
        with self.db:
            existing = self.db.execute("SELECT id, category FROM labels WHERE name=?", (name,)).fetchone()
            if existing and existing[1] != category:
                raise ValueError("La etiqueta ya pertenece a otra categoría")
            if not existing and self.db.execute("SELECT count(*) FROM labels").fetchone()[0] >= self.settings.max_labels:
                raise ValueError("Se alcanzó el límite de etiquetas")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('dimension', ?)", (str(expected),))
            self.db.execute("INSERT OR IGNORE INTO labels(name,category) VALUES (?,?)", (name, category))
            label_id = self.db.execute("SELECT id FROM labels WHERE name=?", (name,)).fetchone()[0]
            saved = [np.frombuffer(row[0], dtype=np.float32) for row in
                     self.db.execute("SELECT vector FROM examples WHERE label_id=?", (label_id,))]
            for vector in vectors:
                if len(saved) >= self.settings.max_examples:
                    break
                if saved and max(float(v @ vector) for v in saved) > 0.995:
                    continue
                self.db.execute("INSERT INTO examples(label_id,vector) VALUES (?,?)", (label_id, vector.tobytes()))
                saved.append(vector)
        self.revision += 1
        return {"id": label_id, "label": name, "category": category, "examples": len(saved)}

    def match(self, vector, category, threshold, margin):
        vector = normalized(vector)
        scores = {}
        for name, raw in self.db.execute(
            "SELECT l.name,e.vector FROM examples e JOIN labels l ON e.label_id=l.id WHERE l.category=?",
            (category,),
        ):
            candidate = np.frombuffer(raw, dtype=np.float32)
            if len(candidate) != len(vector):
                raise ValueError("Dimensión de embedding incompatible")
            scores[name] = max(scores.get(name, -1), float(vector @ candidate))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, None
        name, score = ranked[0]
        score = min(1.0, max(-1.0, score))
        if score < threshold or (len(ranked) > 1 and score - ranked[1][1] < margin):
            return None, score
        return name, score

    def labels(self, after=0, limit=100):
        rows = self.db.execute("""SELECT l.id,l.name,l.category,count(e.id) FROM labels l
            LEFT JOIN examples e ON e.label_id=l.id WHERE l.id>? GROUP BY l.id ORDER BY l.id LIMIT ?""",
                               (after, limit)).fetchall()
        return [{"id": r[0], "label": r[1], "category": r[2], "examples": r[3]} for r in rows]

    def delete(self, label_id):
        with self.db:
            count = self.db.execute("DELETE FROM labels WHERE id=?", (label_id,)).rowcount
        if count:
            self.revision += 1
        return bool(count)

    def event(self, session, track, category, label, confidence, location, kind="sighting"):
        with self.db:
            self.db.execute("INSERT INTO events(timestamp,session,track,category,label,confidence,location,kind) "
                            "VALUES (?,?,?,?,?,?,?,?)", (time.time(), session, track, category, label,
                                                       confidence, json.dumps(location), kind))
            self.db.execute("DELETE FROM events WHERE id <= (SELECT max(id) FROM events)-?",
                            (self.settings.event_limit,))

    def events(self, after=0, limit=100, label=None, since=0, kind=None):
        rows = self.db.execute("""SELECT id,timestamp,session,track,category,label,confidence,location,kind
            FROM events WHERE id>? AND timestamp>=? AND (? IS NULL OR label=?)
            AND (? IS NULL OR kind=?) ORDER BY id LIMIT ?""", (after, since, label, label, kind, kind, limit))
        keys = ["id", "timestamp", "session", "track_id", "category", "label", "confidence", "location", "kind"]
        result = [dict(zip(keys, row)) for row in rows]
        for event in result:
            event["location"] = json.loads(event["location"])
        return result
