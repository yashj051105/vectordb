"""
test_core.py — proof it's correct, not just that it runs.
Run: pytest test_core.py -v
"""

import numpy as np
import pytest

from vectordb.core import VectorDB, FlatIndex, IVFIndex


def brute_force_topk(query, matrix, k):
    qn = query / (np.linalg.norm(query) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    scores = mn @ qn
    return set(np.argsort(-scores)[:k].tolist())


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n, dim = 3000, 32
    # clustered synthetic data, not uniform random -- more realistic geometry
    n_clusters = 15
    centers = rng.normal(scale=3, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    vectors = centers[labels] + rng.normal(scale=0.3, size=(n, dim))
    vectors = vectors.astype(np.float32)
    metadatas = [{"cluster": int(labels[i])} for i in range(n)]
    query = vectors[0] + rng.normal(scale=0.1, size=dim).astype(np.float32)
    return vectors, metadatas, query


def test_flat_index_exact_against_reference(data):
    vectors, _, query = data
    idx = FlatIndex(dim=vectors.shape[1])
    idx.add(vectors)
    rows, _ = idx.search(query, k=10)
    expected = brute_force_topk(query, vectors, 10)
    assert set(rows.tolist()) == expected


def test_ivf_recall_reasonably_high(data):
    vectors, _, query = data
    flat = FlatIndex(dim=vectors.shape[1])
    flat.add(vectors)
    flat_rows, _ = flat.search(query, k=10)

    ivf = IVFIndex(dim=vectors.shape[1], nlist=30, nprobe=10)
    ivf.add(vectors)
    ivf.build(seed=1)
    ivf_rows, _ = ivf.search(query, k=10)

    recall = len(set(flat_rows.tolist()) & set(ivf_rows.tolist())) / len(flat_rows)
    assert recall >= 0.7


def test_ivf_nprobe_curve_is_monotonic_ish(data):
    """More nprobe should never make recall meaningfully worse."""
    vectors, _, query = data
    flat = FlatIndex(dim=vectors.shape[1])
    flat.add(vectors)
    flat_set = set(flat.search(query, k=10)[0].tolist())

    ivf = IVFIndex(dim=vectors.shape[1], nlist=30, nprobe=1)
    ivf.add(vectors)
    ivf.build(seed=1)

    recalls = []
    for nprobe in [1, 5, 15, 30]:
        rows, _ = ivf.search(query, k=10, nprobe=nprobe)
        recalls.append(len(set(rows.tolist()) & flat_set) / len(flat_set))
    assert recalls[-1] >= recalls[0]
    assert recalls[-1] >= 0.95  # nprobe == nlist should recover ~exact


def test_delete_removes_vector_from_results_flat(data):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="flat")
    ids = db.add(vectors, metadatas)
    top = db.search(query, k=1)[0]["id"]
    db.delete([top])
    new_top = db.search(query, k=1)[0]["id"]
    assert new_top != top


def test_delete_removes_vector_from_results_ivf(data):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="ivf", nlist=20, nprobe=10)
    ids = db.add(vectors, metadatas)
    db.build(seed=0)
    top = db.search(query, k=1)[0]["id"]
    db.delete([top])
    new_top = db.search(query, k=1)[0]["id"]
    assert new_top != top


def test_compact_shrinks_flat_index_and_stays_correct(data):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="flat")
    ids = db.add(vectors, metadatas)
    to_delete = ids[:500]
    db.delete(to_delete)
    assert len(db) == len(vectors) - 500
    db.compact()
    assert db.index.vectors.shape[0] == len(vectors) - 500
    # search still returns valid, non-deleted results
    results = db.search(query, k=5)
    assert all(r["id"] not in to_delete for r in results)


def test_compact_rebuilds_ivf_and_stays_searchable(data):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="ivf", nlist=20, nprobe=10)
    ids = db.add(vectors, metadatas)
    db.build(seed=0)
    to_delete = ids[:400]
    db.delete(to_delete)
    db.compact()
    assert db.index.vectors.shape[0] == len(vectors) - 400
    results = db.search(query, k=5)
    assert len(results) > 0
    assert all(r["id"] not in to_delete for r in results)


def test_metadata_filter(data):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="flat")
    db.add(vectors, metadatas)
    target_cluster = metadatas[0]["cluster"]
    results = db.search(query, k=20, where={"cluster": target_cluster})
    assert len(results) > 0
    assert all(r["metadata"]["cluster"] == target_cluster for r in results)


def test_save_load_roundtrip_ivf(data, tmp_path):
    vectors, metadatas, query = data
    db = VectorDB(dim=vectors.shape[1], index_type="ivf", nlist=20, nprobe=8)
    db.add(vectors, metadatas)
    db.build(seed=0)
    path = str(tmp_path / "db")
    db.save(path)
    loaded = VectorDB.load(path)
    r1 = [r["id"] for r in db.search(query, k=5)]
    r2 = [r["id"] for r in loaded.search(query, k=5)]
    assert r1 == r2
