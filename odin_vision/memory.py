"""Memoria vectorial y episódica transaccional, acotada y persistente en SQLite."""
import json
import sqlite3
import time
from itertools import pairwise

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
            CREATE INDEX IF NOT EXISTS event_label_id ON events(label, id, timestamp);
            DROP INDEX IF EXISTS event_label_time;
            CREATE INDEX IF NOT EXISTS examples_label ON examples(label_id);
            CREATE INDEX IF NOT EXISTS labels_category ON labels(category);
        """)
        old = self.db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()
        if old and old[0] != fingerprint:
            self.db.close()
            raise ValueError("El encoder ha cambiado: use otro ODIN_DATA_DIR o exporte su memoria")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('encoder', ?)", (fingerprint,))
        self.db.commit()
        self.revision = 0
        self._galleries = {}
        self._data_version = self.db.execute("PRAGMA data_version").fetchone()[0]

    def close(self):
        self._galleries.clear()
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
        self._galleries.pop(category, None)
        return {"id": label_id, "label": name, "category": category, "examples": len(saved)}

    def _gallery(self, category):
        # data_version detecta commits de otras conexiones; learn/delete invalidan los propios.
        version = self.db.execute("PRAGMA data_version").fetchone()[0]
        if version != self._data_version:
            self._galleries.clear()
            self._data_version = version
        if category in self._galleries:
            return self._galleries[category]
        names, indices, vectors = {}, [], []
        cursor = self.db.execute(
            "SELECT e.id,l.name,e.vector FROM examples e JOIN labels l ON e.label_id=l.id WHERE l.category=?",
            (category,),
        )
        rows = cursor.fetchmany(65)
        if len(rows) == 65 and any(a[0] > b[0] for a, b in pairwise(rows)):
            # Leer los BLOB por id si el índice de etiquetas salta entre páginas alejadas.
            cursor.close()
            rows = self.db.execute(
                "SELECT g.id,g.name,e.vector FROM "
                "(SELECT e.id,l.name FROM examples e JOIN labels l ON e.label_id=l.id "
                "WHERE l.category=? ORDER BY e.id LIMIT -1) g "
                "CROSS JOIN examples e ON e.id=g.id", (category,),
            ).fetchall()
        else:
            rows.extend(cursor.fetchall())
            cursor.close()
        # Preservar el orden original de ejemplos para desempatar. Ordenar referencias
        # aquí evita que SQLite copie todos los BLOB a una tabla temporal de ordenación.
        rows.sort(key=lambda row: row[0])
        for _, name, raw in rows:
            indices.append(names.setdefault(name, len(names)))
            vectors.append(raw)
        if not vectors:
            # No acumular entradas para categorías arbitrarias que no existen.
            return None
        if any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("Dimensión de embedding incompatible")
        # Una unión evita crear un ndarray por ejemplo antes de copiar la matriz final.
        matrix = np.frombuffer(bytearray().join(vectors), dtype=np.float32).reshape(len(vectors), -1)
        gallery = (tuple(names), np.asarray(indices), matrix)
        self._galleries[category] = gallery
        return gallery

    def has_examples(self, category):
        return self._gallery(category) is not None

    def match(self, vector, category, threshold, margin):
        vector = normalized(vector)
        gallery = self._gallery(category)
        if gallery is None:
            return None, None
        matrix = gallery[2]
        if matrix.shape[1] != len(vector):
            raise ValueError("Dimensión de embedding incompatible")
        similarities = np.einsum("ij,j->i", matrix, vector, optimize=False)
        return self._rank(vector, gallery, similarities, threshold, margin)

    def match_many(self, vectors, category, threshold, margin):
        vectors = [normalized(vector) for vector in vectors]
        if not vectors:
            return []
        gallery = self._gallery(category)
        if gallery is None:
            return [(None, None)] * len(vectors)
        matrix = gallery[2]
        if any(len(vector) != matrix.shape[1] for vector in vectors):
            raise ValueError("Dimensión de embedding incompatible")
        if len(vectors) == 1 and len(matrix) < 16000:
            similarities = np.einsum("ij,j->i", matrix, vectors[0], optimize=False)
            return [self._rank(vectors[0], gallery, similarities, threshold, margin)]
        matches = []
        # Acotar la matriz temporal aunque haya muchos objetos en la escena.
        for start in range(0, len(vectors), 32):
            queries = np.stack(vectors[start:start+32])
            # Evitar el coste fijo de BLAS en lotes medianos; galerías grandes sí lo amortizan.
            similarities = (queries @ matrix.T if len(matrix) >= 8000 or len(queries) * len(matrix) >= 24000
                            else np.einsum("ij,kj->ki", matrix, queries, optimize=False))
            matches.extend(self._rank(vector, gallery, scores, threshold, margin)
                           for vector, scores in zip(queries, similarities))
        return matches

    @staticmethod
    def _rank(vector, gallery, similarities, threshold, margin):
        names, indices, matrix = gallery
        if len(names) == 1:
            cutoff = max(-1.0, float(similarities.max()))
            tolerance = 4 * len(vector) * np.finfo(np.float32).eps
            score = max((float(vector @ matrix[index]) for index in
                         np.flatnonzero(similarities >= cutoff-tolerance)), default=-1.0)
            score = min(1.0, max(-1.0, score))
            return (None if score < threshold else names[0]), score
        # El segundo máximo por etiqueta es el mejor ejemplo ajeno a la etiqueta ganadora.
        best_label = indices[int(np.argmax(similarities))]
        cutoff = similarities[indices != best_label].max(initial=-1.0)
        # La reducción vectorizada puede redondear distinto a dot. Refinar todos los
        # candidatos cercanos a los dos mejores conserva empates, umbral y margen.
        tolerance = 4 * len(vector) * np.finfo(np.float32).eps
        # Dos sentinelas conservan los desempates originales incluso si todos los
        # scores quedan en -1, sin crear una entrada para cada etiqueta de la galería.
        scores = {0: -1.0, 1: -1.0}
        for index in np.flatnonzero(similarities >= cutoff-tolerance):
            label_index = int(indices[index])
            scores[label_index] = max(scores.get(label_index, -1.0), float(vector @ matrix[index]))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        label_index, score = ranked[0]
        score = min(1.0, max(-1.0, score))
        if score < threshold or (len(ranked) > 1 and score - ranked[1][1] < margin):
            return None, score
        return names[label_index], score

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
            self._galleries.clear()
        return bool(count)

    def event(self, session, track, category, label, confidence, location, kind="sighting"):
        self.event_many([(time.time(), session, track, category, label, confidence, location, kind)])

    def event_many(self, events):
        """Persiste los eventos de un frame juntos, en orden, y aplica retención una sola vez."""
        rows = [(timestamp, session, track, category, label, confidence, json.dumps(location), kind)
                for timestamp, session, track, category, label, confidence, location, kind in events]
        if not rows:
            return
        with self.db:
            self.db.executemany("INSERT INTO events(timestamp,session,track,category,label,confidence,location,kind) "
                                "VALUES (?,?,?,?,?,?,?,?)", rows)
            self.db.execute("DELETE FROM events WHERE id <= (SELECT max(id) FROM events)-?",
                            (self.settings.event_limit,))

    def events(self, after=0, limit=100, label=None, since=0, kind=None):
        clauses = ["id>?", "timestamp>=?"]
        parameters = [after, since]
        if label is not None:
            clauses.append("label=?")
            parameters.append(label)
        if kind is not None:
            clauses.append("kind=?")
            parameters.append(kind)
        rows = self.db.execute(
            "SELECT id,timestamp,session,track,category,label,confidence,location,kind "
            "FROM events WHERE " + " AND ".join(clauses) + " ORDER BY id LIMIT ?",
            (*parameters, limit),
        )
        keys = ["id", "timestamp", "session", "track_id", "category", "label", "confidence", "location", "kind"]
        result = [dict(zip(keys, row)) for row in rows]
        for event in result:
            event["location"] = json.loads(event["location"])
        return result
