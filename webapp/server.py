"""
webapp/server.py — a real browser interface for the vector DB, using only
Python's standard library (http.server). No Flask, no Django — the point
of this project is showing what's underneath, and a web framework would
just be one more black box on top.

Run: python webapp/server.py
Then open http://localhost:8000
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import generate_corpus
from vectordb.embeddings import TfidfEmbedder
from vectordb.core import VectorDB

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

print("Building demo index (5,000 generated support tickets)...")
_docs, _labels = generate_corpus(n_per_topic=500, seed=42)
_embedder = TfidfEmbedder(n_components=64)
_vectors = _embedder.fit(_docs)

FLAT_DB = VectorDB(dim=_vectors.shape[1], metric="cosine", index_type="flat")
FLAT_DB.add(_vectors, [{"text": d, "topic": l} for d, l in zip(_docs, _labels)])

IVF_DB = VectorDB(dim=_vectors.shape[1], metric="cosine", index_type="ivf", nlist=48, nprobe=6)
IVF_DB.add(_vectors, [{"text": d, "topic": l} for d, l in zip(_docs, _labels)])
IVF_DB.build(seed=0)

TOPICS = sorted(set(_labels))
print(f"Ready: {len(_docs)} documents, {len(TOPICS)} topics, dim={_vectors.shape[1]}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console clean

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path == "/api/meta":
            self._send_json({"n_docs": len(_docs), "topics": TOPICS, "dim": int(_vectors.shape[1])})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/api/search":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "bad request"}, status=400)
            return

        query = (payload.get("query") or "").strip()
        mode = payload.get("mode", "flat")
        nprobe = int(payload.get("nprobe", 6))
        k = int(payload.get("k", 6))
        topic = payload.get("topic") or None

        if not query:
            self._send_json({"results": [], "elapsed_ms": 0})
            return

        qvec = _embedder.transform([query])[0]
        where = {"topic": topic} if topic else None

        t0 = time.perf_counter()
        if mode == "ivf":
            results = IVF_DB.search(qvec, k=k, nprobe=nprobe, where=where)
        else:
            results = FLAT_DB.search(qvec, k=k, where=where)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._send_json({
            "results": [
                {"score": r["score"], "text": r["metadata"]["text"], "topic": r["metadata"]["topic"]}
                for r in results
            ],
            "elapsed_ms": elapsed_ms,
        })


def main():
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\nOpen http://localhost:{port} in your browser.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
