"""
benchmark.py — the actual assignment. Everything else is scaffolding.

1. Generate a real 5,000-text corpus, embed it (TF-IDF+SVD).
2. Scale to >=50,000 vectors: each of the 5,000 base embeddings is
   replicated 10x with small gaussian jitter. This keeps the real-text
   geometry (clusters stay lumpy, not uniform-random) while reaching the
   scale the exact-index requirement asks for. Documented, not hidden.
3. Build a FlatIndex over all 50,000 — this is ground truth.
4. Build an IVFIndex over the same 50,000 vectors.
5. Generate 500 query vectors (held-out texts, not in the indexed set),
   compute their exact top-10 neighbors via FlatIndex.
6. For 6 nprobe settings, measure recall@10 and queries/sec (QPS).
   Print a table and save a matplotlib plot of the accuracy/speed curve.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corpus import generate_corpus
from vectordb.embeddings import TfidfEmbedder
from vectordb.core import VectorDB


N_BASE = 5000
REPLICAS = 10          # -> 50,000 vectors total
N_QUERIES = 500
TOP_K = 10
NLIST = 500
NPROBE_SETTINGS = [1, 2, 3, 5, 8, 15, 30, 60, 120]
JITTER_SCALE = 0.04
SEED = 42


def build_scaled_corpus(embedder: TfidfEmbedder):
    docs, labels = generate_corpus(n_per_topic=N_BASE // 10, seed=SEED)
    base_vectors = embedder.fit(docs)

    rng = np.random.default_rng(SEED)
    n = base_vectors.shape[0]
    idx = np.repeat(np.arange(n), REPLICAS)
    noise = rng.normal(scale=JITTER_SCALE, size=(len(idx), base_vectors.shape[1])).astype(np.float32)
    scaled_vectors = base_vectors[idx] + noise
    scaled_meta = [{"text": docs[i], "topic": labels[i], "base_doc": int(i)} for i in idx]
    return scaled_vectors.astype(np.float32), scaled_meta, docs, labels


def build_queries(embedder: TfidfEmbedder, seed=SEED + 1):
    docs, labels = generate_corpus(n_per_topic=max(1, N_QUERIES // 10), seed=seed)
    docs, labels = docs[:N_QUERIES], labels[:N_QUERIES]
    vectors = embedder.transform(docs)
    return vectors.astype(np.float32), docs, labels


def recall_at_k(ivf_ids, flat_ids):
    a, b = set(ivf_ids), set(flat_ids)
    if not b:
        return 1.0
    return len(a & b) / len(b)


def main():
    print(f"Generating and embedding {N_BASE} real texts...")
    embedder = TfidfEmbedder(n_components=96)
    scaled_vectors, scaled_meta, base_docs, base_labels = build_scaled_corpus(embedder)
    n_total = scaled_vectors.shape[0]
    print(f"Scaled to {n_total} vectors ({N_BASE} base docs x {REPLICAS} jittered replicas)")

    print("\nBuilding FlatIndex (ground truth, exact)...")
    flat_db = VectorDB(dim=scaled_vectors.shape[1], metric="cosine", index_type="flat")
    flat_db.add(scaled_vectors, scaled_meta)

    print("Building IVFIndex (approximate)...")
    ivf_db = VectorDB(dim=scaled_vectors.shape[1], metric="cosine", index_type="ivf",
                       nlist=NLIST, nprobe=NPROBE_SETTINGS[0])
    ivf_db.add(scaled_vectors, scaled_meta)
    t0 = time.perf_counter()
    ivf_db.build(iters=12, seed=0)
    print(f"IVF build time: {(time.perf_counter() - t0)*1000:.1f} ms (nlist={NLIST})")

    print(f"\nGenerating {N_QUERIES} held-out queries + exact ground truth (FlatIndex)...")
    query_vectors, query_docs, query_labels = build_queries(embedder)

    ground_truth = []
    t0 = time.perf_counter()
    for qv in query_vectors:
        results = flat_db.search(qv, k=TOP_K)
        ground_truth.append([r["id"] for r in results])
    flat_elapsed = time.perf_counter() - t0
    flat_qps = N_QUERIES / flat_elapsed
    print(f"FlatIndex: {flat_elapsed:.2f}s total, {flat_qps:.1f} queries/sec (exact, ground truth)")

    print(f"\n{'nprobe':>8} | {'recall@10':>10} | {'QPS':>10} | {'ms/query':>10}")
    print("-" * 48)
    curve = []
    for nprobe in NPROBE_SETTINGS:
        recalls = []
        t0 = time.perf_counter()
        for qv, gt in zip(query_vectors, ground_truth):
            results = ivf_db.search(qv, k=TOP_K, nprobe=nprobe)
            got = [r["id"] for r in results]
            recalls.append(recall_at_k(got, gt))
        elapsed = time.perf_counter() - t0
        qps = N_QUERIES / elapsed
        mean_recall = float(np.mean(recalls))
        curve.append((nprobe, mean_recall, qps))
        print(f"{nprobe:>8} | {mean_recall:>10.3f} | {qps:>10.1f} | {1000*elapsed/N_QUERIES:>10.3f}")

    print(f"\nFor reference, exact search (FlatIndex): {flat_qps:.1f} QPS at recall=1.000")

    nprobes = [c[0] for c in curve]
    recalls = [c[1] for c in curve]
    qpss = [c[2] for c in curve]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(nprobes, recalls, marker="o", color="#2b6cb0")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="exact (FlatIndex)")
    axes[0].set_xlabel("nprobe (clusters searched)")
    axes[0].set_ylabel("recall@10 vs exact search")
    axes[0].set_title("Accuracy vs. nprobe")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(recalls, qpss, marker="o", color="#c0392b")
    axes[1].axhline(flat_qps, color="gray", linestyle="--", linewidth=1, label="exact (FlatIndex) QPS")
    axes[1].set_xlabel("recall@10")
    axes[1].set_ylabel("queries per second")
    axes[1].set_title("The actual trade-off: accuracy vs. speed")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"IVF speed/accuracy curve — {n_total:,} vectors, nlist={NLIST}", fontsize=12)
    fig.tight_layout()
    fig.savefig("benchmark_plot.png", dpi=140)
    print("\nSaved plot to benchmark_plot.png")


if __name__ == "__main__":
    main()
