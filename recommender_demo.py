from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "scholars_course_recommender.joblib"
DATASET_LABEL = "ScholarS course recommendation prototype (substitute ratings)"
TOP_K = 5


def load_demo_ratings():
    return Dataset.load_builtin("ml-100k", prompt=False)


def hit_rate_at_k(recommended: list[str], held_out: str, k: int = TOP_K) -> float:
    return float(held_out in recommended[:k])


def ndcg_at_k(recommended: list[str], held_out: str, k: int = TOP_K) -> float:
    try:
        rank = recommended[:k].index(held_out) + 1
    except ValueError:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def recommend_popularity(trainset, user_id: str, k: int = TOP_K) -> list[str]:
    seen = {item_id for item_id, _ in trainset.ur[trainset.to_inner_uid(user_id)]}
    popular = [item_id for item_id, _ in Counter(
        trainset.to_raw_iid(item_inner_id)
        for _, item_inner_id, _ in trainset.all_ratings()
    ).most_common()]
    return [item_id for item_id in popular if trainset.to_inner_iid(item_id) not in seen][:k]


def recommend_svd(model: SVD, trainset, user_id: str, k: int = TOP_K) -> list[str]:
    seen_inner = {item_inner_id for item_inner_id, _ in trainset.ur[trainset.to_inner_uid(user_id)]}
    candidates = [
        trainset.to_raw_iid(item_inner_id)
        for item_inner_id in trainset.all_items()
        if item_inner_id not in seen_inner
    ]
    scored = sorted(
        ((item_id, model.predict(user_id, item_id).est) for item_id in candidates),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [item_id for item_id, _ in scored[:k]]


def evaluate() -> dict[str, object]:
    print("Loading substitute learner-course ratings...", flush=True)
    data = load_demo_ratings()
    full_trainset = data.build_full_trainset()
    total_ratings = full_trainset.n_users * full_trainset.n_items
    sparsity = 1 - (full_trainset.n_ratings / total_ratings)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    print("Training SVD collaborative-filtering model...", flush=True)
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)

    held_out_by_user = defaultdict(list)
    for user_id, item_id, _ in testset:
        if user_id in trainset._raw2inner_id_users:
            held_out_by_user[user_id].append(item_id)

    popularity_hits = []
    popularity_ndcgs = []
    svd_hits = []
    svd_ndcgs = []
    evaluated = 0

    for user_id, held_out_items in held_out_by_user.items():
        popularity = recommend_popularity(trainset, user_id)
        svd = recommend_svd(model, trainset, user_id)
        for item_id in held_out_items:
            popularity_hits.append(hit_rate_at_k(popularity, item_id))
            popularity_ndcgs.append(ndcg_at_k(popularity, item_id))
            svd_hits.append(hit_rate_at_k(svd, item_id))
            svd_ndcgs.append(ndcg_at_k(svd, item_id))
            evaluated += 1

    print("Training final model and saving artifact...", flush=True)
    full_model = SVD(n_factors=100, n_epochs=20, random_state=42).fit(full_trainset)
    popularity = [
        item_id for item_id, _ in Counter(
            full_trainset.to_raw_iid(item_inner_id)
            for _, item_inner_id, _ in full_trainset.all_ratings()
        ).most_common()
    ]
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": full_model, "trainset": full_trainset, "popularity": popularity},
        ARTIFACT_PATH,
    )

    return {
        "dataset": DATASET_LABEL,
        "users": full_trainset.n_users,
        "items": full_trainset.n_items,
        "ratings": full_trainset.n_ratings,
        "interaction_matrix_sparsity": float(sparsity),
        "evaluated_test_rows": evaluated,
        "popularity_hit_rate_at_5": float(np.mean(popularity_hits)),
        "popularity_ndcg_at_5": float(np.mean(popularity_ndcgs)),
        "svd_hit_rate_at_5": float(np.mean(svd_hits)),
        "svd_ndcg_at_5": float(np.mean(svd_ndcgs)),
        "artifact": str(ARTIFACT_PATH),
    }


if __name__ == "__main__":
    metrics = evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value}")
