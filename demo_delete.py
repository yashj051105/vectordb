"""
demo_delete.py — shows insert / search / delete / compact end to end,
on both index types, so the delete behavior described in core.py's
docstring is visible, not just asserted in a test.
"""

import numpy as np
from vectordb.core import VectorDB


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run(index_type):
    section(f"{index_type.upper()} INDEX — insert, search, delete, compact")

    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(200, 12)).astype(np.float32)
    metadatas = [{"label": f"item-{i}"} for i in range(200)]

    kwargs = {"nlist": 10, "nprobe": 5} if index_type == "ivf" else {}
    db = VectorDB(dim=12, index_type=index_type, **kwargs)
    ids = db.add(vectors, metadatas)
    if index_type == "ivf":
        db.build(seed=0)
    print(f"Inserted {len(db)} vectors.")

    query = vectors[10]
    top = db.search(query, k=3)
    print(f"Search before delete, top match: id={top[0]['id']} label={top[0]['metadata']['label']}")

    target_id = ids[10]
    db.delete([target_id])
    print(f"Deleted id={target_id}. len(db) is now {len(db)} (row still occupies memory until compact()).")

    top_after = db.search(query, k=3)
    print(f"Search after delete, top match: id={top_after[0]['id']} label={top_after[0]['metadata']['label']}")
    assert top_after[0]["id"] != target_id, "deleted vector should never be returned again"
    print("Confirmed: deleted vector is never returned, even before compaction.")

    db.compact()
    print(f"Ran compact(). Physical row count is now {db.index.vectors.shape[0]}.")
    top_final = db.search(query, k=3)
    print(f"Search after compact, top match: id={top_final[0]['id']} label={top_final[0]['metadata']['label']}")


if __name__ == "__main__":
    run("flat")
    run("ivf")
    print("\nSee vectordb/core.py's module docstring for why IVF's compact()")
    print("has to fully re-cluster, while FlatIndex's compact() doesn't.")
