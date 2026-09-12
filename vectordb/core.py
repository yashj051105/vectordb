"""
core.py — the actual vector database. numpy only, nothing above it.

Two indexes:

  FlatIndex   — exact, brute-force. O(n) per query. This is ground truth;
                every claim the approximate index makes is checked against it.

  IVFIndex    — approximate, via inverted-file clustering (k-means
                partitioning + probe-nearest-clusters search). The
                accuracy/speed knob is `nprobe`: how many of the nlist
                clusters get searched per query. nprobe=nlist is
                equivalent to exact search (slow); nprobe=1 is fastest
                and least accurate. See benchmark.py for the curve.

Insert / search / delete:
  - Insert: append to the vector array; for IVF, assign the new vector
    to its nearest EXISTING centroid (no full rebuild).
  - Search: as above.
  - Delete: THIS IS THE HARD ONE, and here is the honest account of why.
    A vector's position in a numpy array is what every index structure
    (IVF buckets, in particular) references internally. Physically
    removing a row means every row after it shifts down by one, which
    would silently invalidate every bucket's stored row numbers unless
    they are all rebuilt. That's an O(n) operation per delete if done
    naively on every call — unacceptable for a database.
    The standard real-world fix, used here, is TOMBSTONING: mark a row
    "dead" in a boolean mask, skip dead rows during search, and defer
    physical cleanup to an explicit, batched compact() call. This makes
    delete() O(1), but has two real costs this project does not hide:
      1. Deleted vectors still occupy memory and get scanned (and
         discarded) during search until compact() runs — search gets
         slower over time if you delete a lot without compacting.
      2. For IVFIndex specifically, compact() must fully re-cluster,
         because row numbers inside every bucket become invalid the
         moment any row physically moves. There is no cheap partial
         fix for that — it is a full rebuild, same cost as building
         the index the first time.
"""

import json
import os
import time
import numpy as np


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return mat / norms


def _scores(query: np.ndarray, matrix: np.ndarray, metric: str) -> np.ndarray:
    """Higher = more similar, for either metric, so top-k logic is shared."""
    if matrix.shape[0] == 0:
        return np.array([])
    if metric == "cosine":
        q = _normalize(query.reshape(1, -1))
        m = _normalize(matrix)
        return (m @ q.T).ravel()
    elif metric == "l2":
        diffs = matrix - query.reshape(1, -1)
        return -np.einsum("ij,ij->i", diffs, diffs)
    raise ValueError(f"unknown metric: {metric}")


def _topk(scores: np.ndarray, k: int):
    if len(scores) == 0:
        return np.array([], dtype=int), np.array([])
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


class FlatIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)

    def add(self, vectors: np.ndarray):
        start = self.vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vectors]) if start else vectors.copy()
        self.alive = np.concatenate([self.alive, np.ones(len(vectors), dtype=bool)])
        return list(range(start, start + len(vectors)))

    def search(self, query: np.ndarray, k: int):
        if not self.alive.any():
            return np.array([], dtype=int), np.array([])
        alive_rows = np.nonzero(self.alive)[0]
        scores = _scores(query, self.vectors[alive_rows], self.metric)
        local_idx, local_scores = _topk(scores, k)
        return alive_rows[local_idx], local_scores

    def delete(self, rows):
        self.alive[list(rows)] = False

    def compact(self):
        keep = np.nonzero(self.alive)[0]
        mapping = {int(old): new for new, old in enumerate(keep)}
        self.vectors = self.vectors[keep]
        self.alive = np.ones(len(keep), dtype=bool)
        return mapping


def _kmeans(data: np.ndarray, n_clusters: int, iters: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    n_clusters = max(1, min(n_clusters, n))

    centers = np.empty((n_clusters, data.shape[1]), dtype=data.dtype)
    centers[0] = data[rng.integers(n)]
    closest_sq = np.full(n, np.inf)
    for c in range(1, n_clusters):
        diffs = data - centers[c - 1]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        closest_sq = np.minimum(closest_sq, d2)
        probs = closest_sq / (closest_sq.sum() + 1e-12)
        centers[c] = data[rng.choice(n, p=probs)]

    assignments = np.zeros(n, dtype=int)
    data_sq = np.einsum("ij,ij->i", data, data)  # ||x||^2 per row, computed once
    for it in range(iters):
        # ||x - c||^2 = ||x||^2 - 2 x.c^T + ||c||^2, avoids an (n, k, dim) tensor
        center_sq = np.einsum("ij,ij->i", centers, centers)
        d = data_sq[:, None] - 2 * (data @ centers.T) + center_sq[None, :]
        new_assign = d.argmin(axis=1)
        if it > 0 and np.array_equal(new_assign, assignments):
            break
        assignments = new_assign
        for c in range(n_clusters):
            members = data[assignments == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return centers, assignments


class IVFIndex:
    def __init__(self, dim: int, metric: str = "cosine", nlist: int = 32, nprobe: int = 4):
        self.dim = dim
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.centers = None
        self.assignments = np.zeros(0, dtype=int)
        self.buckets = {}

    def add(self, vectors: np.ndarray):
        start = self.vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vectors]) if start else vectors.copy()
        self.alive = np.concatenate([self.alive, np.ones(len(vectors), dtype=bool)])
        new_rows = list(range(start, start + len(vectors)))

        if self.centers is not None:
            # online insert: assign to nearest EXISTING centroid, no rebuild.
            # (honest cost: cluster quality slowly degrades between rebuilds
            # if many inserts land far from any existing centroid.)
            for i, row in enumerate(new_rows):
                c_scores = _scores(vectors[i], self.centers, self.metric)
                cluster = int(np.argmax(c_scores))
                self.assignments = np.append(self.assignments, cluster)
                self.buckets.setdefault(cluster, []).append(row)
        return new_rows

    def build(self, iters: int = 15, seed: int = 0):
        """(Re)cluster everything currently alive. Also the only way to
        recover from deletions — see module docstring."""
        alive_rows = np.nonzero(self.alive)[0]
        if len(alive_rows) == 0:
            return
        data = self.vectors[alive_rows]
        self.centers, local_assign = _kmeans(data, self.nlist, iters=iters, seed=seed)
        self.assignments = np.zeros(self.vectors.shape[0], dtype=int) - 1
        self.buckets = {}
        for local_row, cluster in zip(alive_rows, local_assign):
            self.assignments[local_row] = cluster
            self.buckets.setdefault(int(cluster), []).append(int(local_row))

    def search(self, query: np.ndarray, k: int, nprobe: int = None):
        if self.centers is None:
            return np.array([], dtype=int), np.array([])
        nprobe = nprobe or self.nprobe
        center_scores = _scores(query, self.centers, self.metric)
        probe_clusters, _ = _topk(center_scores, min(nprobe, len(self.centers)))

        candidate_rows = []
        for c in probe_clusters:
            candidate_rows.extend(self.buckets.get(int(c), []))
        if not candidate_rows:
            return np.array([], dtype=int), np.array([])
        candidate_rows = np.array(candidate_rows)
        alive_mask = self.alive[candidate_rows]
        candidate_rows = candidate_rows[alive_mask]
        if len(candidate_rows) == 0:
            return np.array([], dtype=int), np.array([])
        sub_scores = _scores(query, self.vectors[candidate_rows], self.metric)
        local_idx, scores = _topk(sub_scores, k)
        return candidate_rows[local_idx], scores

    def delete(self, rows):
        """O(1) tombstone. Buckets keep stale row numbers; search() filters
        them via self.alive. Row numbers stay valid until compact()."""
        self.alive[list(rows)] = False

    def compact(self, iters: int = 15, seed: int = 0):
        """The expensive path: physically drop dead rows and re-cluster
        from scratch, because row numbers inside every bucket become
        invalid the instant any row shifts position."""
        keep = np.nonzero(self.alive)[0]
        mapping = {int(old): new for new, old in enumerate(keep)}
        self.vectors = self.vectors[keep]
        self.alive = np.ones(len(keep), dtype=bool)
        self.build(iters=iters, seed=seed)
        return mapping


class VectorDB:
    def __init__(self, dim: int, metric: str = "cosine", index_type: str = "flat",
                 nlist: int = 32, nprobe: int = 4):
        self.dim = dim
        self.metric = metric
        self.index_type = index_type
        if index_type == "flat":
            self.index = FlatIndex(dim, metric)
        elif index_type == "ivf":
            self.index = IVFIndex(dim, metric, nlist=nlist, nprobe=nprobe)
        else:
            raise ValueError("index_type must be 'flat' or 'ivf'")
        self.row_to_meta = {}
        self.row_to_id = {}
        self.id_to_row = {}
        self._next_id = 0

    def add(self, vectors: np.ndarray, metadatas: list, ids: list = None):
        vectors = vectors.astype(np.float32)
        rows = self.index.add(vectors)
        if ids is None:
            ids = list(range(self._next_id, self._next_id + len(metadatas)))
        for row, meta, _id in zip(rows, metadatas, ids):
            self.row_to_meta[row] = meta
            self.row_to_id[row] = _id
            self.id_to_row[_id] = row
        self._next_id = max(self._next_id, max(ids) + 1) if ids else self._next_id
        return ids

    def build(self, **kwargs):
        if isinstance(self.index, IVFIndex):
            self.index.build(**kwargs)

    def search(self, query_vector: np.ndarray, k: int = 5, where: dict = None, **kwargs):
        fetch_k = k if where is None else max(k * 10, k + 20)
        rows, scores = self.index.search(query_vector.astype(np.float32), fetch_k, **kwargs)
        results = []
        for row, score in zip(rows, scores):
            meta = self.row_to_meta.get(int(row), {})
            if where and not all(meta.get(mk) == mv for mk, mv in where.items()):
                continue
            results.append({"id": self.row_to_id.get(int(row)), "score": float(score), "metadata": meta})
            if len(results) >= k:
                break
        return results

    def delete(self, ids: list):
        rows = [self.id_to_row[i] for i in ids if i in self.id_to_row]
        self.index.delete(rows)
        for i in ids:
            row = self.id_to_row.pop(i, None)
            if row is not None:
                self.row_to_meta.pop(row, None)
                self.row_to_id.pop(row, None)
        return len(rows)

    def compact(self, **kwargs):
        mapping = self.index.compact(**kwargs)
        new_meta, new_id, new_id_to_row = {}, {}, {}
        for old_row, new_row in mapping.items():
            if old_row in self.row_to_meta:
                new_meta[new_row] = self.row_to_meta[old_row]
                _id = self.row_to_id[old_row]
                new_id[new_row] = _id
                new_id_to_row[_id] = new_row
        self.row_to_meta, self.row_to_id, self.id_to_row = new_meta, new_id, new_id_to_row

    def __len__(self):
        return int(self.index.alive.sum()) if hasattr(self.index, "alive") else len(self.row_to_meta)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self.index.vectors)
        np.save(os.path.join(path, "alive.npy"), self.index.alive)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "dim": self.dim, "metric": self.metric, "index_type": self.index_type,
                "nlist": getattr(self.index, "nlist", None),
                "nprobe": getattr(self.index, "nprobe", None),
                "row_to_meta": {str(k): v for k, v in self.row_to_meta.items()},
                "row_to_id": {str(k): v for k, v in self.row_to_id.items()},
                "next_id": self._next_id,
            }, f)
        if isinstance(self.index, IVFIndex) and self.index.centers is not None:
            np.save(os.path.join(path, "centers.npy"), self.index.centers)
            np.save(os.path.join(path, "assignments.npy"), self.index.assignments)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        db = cls(meta["dim"], meta["metric"], meta["index_type"],
                  nlist=meta.get("nlist") or 32, nprobe=meta.get("nprobe") or 4)
        db.index.vectors = np.load(os.path.join(path, "vectors.npy"))
        db.index.alive = np.load(os.path.join(path, "alive.npy"))
        db.row_to_meta = {int(k): v for k, v in meta["row_to_meta"].items()}
        db.row_to_id = {int(k): v for k, v in meta["row_to_id"].items()}
        db.id_to_row = {v: k for k, v in db.row_to_id.items()}
        db._next_id = meta["next_id"]
        centers_path = os.path.join(path, "centers.npy")
        if isinstance(db.index, IVFIndex) and os.path.exists(centers_path):
            db.index.centers = np.load(centers_path)
            db.index.assignments = np.load(os.path.join(path, "assignments.npy"))
            db.index.buckets = {}
            for row, cluster in enumerate(db.index.assignments):
                if cluster >= 0:
                    db.index.buckets.setdefault(int(cluster), []).append(row)
        return db


def timed_search(db: VectorDB, query_vector: np.ndarray, k: int = 5, **kwargs):
    t0 = time.perf_counter()
    results = db.search(query_vector, k, **kwargs)
    return results, (time.perf_counter() - t0) * 1000
