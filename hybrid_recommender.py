from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_CACHE = Path.home() / ".surprise_data" / "ml-100k" / "ml-100k"
ARTIFACT_PATH = ROOT / "artifacts" / "scholars_hybrid_recommender.joblib"
TOP_K = 5


def load_metadata() -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["item_id", "title", "release_date", "video_release_date", "url"]
    genre_columns = [f"genre_{index}" for index in range(19)]
    items = pd.read_csv(
        DATA_CACHE / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=columns + genre_columns,
        usecols=[0, 1, *range(5, 24)],
    )
    return items, items[genre_columns].to_numpy(dtype=float)


def load_ratings():
    return Dataset.load_builtin("ml-100k", prompt=False)


def top_k_metrics(recommendations: list[str], held_out: list[str], k: int = TOP_K) -> tuple[float, float]:
    ranks = [recommendations[:k].index(item) + 1 for item in held_out if item in recommendations[:k]]
    if not ranks:
        return 0.0, 0.0
    return 1.0, float(max(1.0 / np.log2(rank + 1) for rank in ranks))


def item_popularity(trainset) -> list[str]:
    counts = Counter(
        trainset.to_raw_iid(item_inner_id)
        for _, item_inner_id, _ in trainset.all_ratings()
    )
    return [item_id for item_id, _ in counts.most_common()]


def candidate_items(trainset, user_id: str) -> list[str]:
    if user_id not in trainset._raw2inner_id_users:
        return [trainset.to_raw_iid(item_id) for item_id in trainset.all_items()]
    seen = {item_id for item_id, _ in trainset.ur[trainset.to_inner_uid(user_id)]}
    return [
        trainset.to_raw_iid(item_id)
        for item_id in trainset.all_items()
        if item_id not in seen
    ]


def minmax(values: np.ndarray) -> np.ndarray:
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def recommend_hybrid(
    model: SVD,
    trainset,
    item_features: dict[str, np.ndarray],
    user_id: str,
    popularity: list[str],
    k: int = TOP_K,
    collaborative_weight: float = 0.7,
) -> list[str]:
    candidates = candidate_items(trainset, user_id)
    if user_id not in trainset._raw2inner_id_users:
        return popularity[:k]

    user_ratings = trainset.ur[trainset.to_inner_uid(user_id)]
    profile_weights = np.array([max(rating - 3.0, 0.1) for _, rating in user_ratings])
    profile_vectors = np.array([item_features[trainset.to_raw_iid(item)] for item, _ in user_ratings])
    profile = np.average(profile_vectors, axis=0, weights=profile_weights)
    candidate_features = np.array([item_features[item_id] for item_id in candidates])
    profile_norm = np.linalg.norm(profile)
    candidate_norms = np.linalg.norm(candidate_features, axis=1)
    denominator = profile_norm * candidate_norms
    content_scores = np.divide(
        candidate_features @ profile,
        denominator,
        out=np.zeros(len(candidates), dtype=float),
        where=denominator != 0,
    )
    user_inner_id = trainset.to_inner_uid(user_id)
    candidate_inner_ids = np.array([trainset.to_inner_iid(item_id) for item_id in candidates])
    collaborative_scores = (
        model.trainset.global_mean
        + model.bu[user_inner_id]
        + model.bi[candidate_inner_ids]
        + model.pu[user_inner_id] @ model.qi[candidate_inner_ids].T
    )
    scores = collaborative_weight * minmax(collaborative_scores) + (1 - collaborative_weight) * minmax(content_scores)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[:k]]


def evaluate() -> dict[str, object]:
    print("Loading ratings and course-content stand-in features...", flush=True)
    data = load_ratings()
    items, feature_matrix = load_metadata()
    item_features = {
        str(row.item_id): feature_matrix[index]
        for index, row in items.iterrows()
    }
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    print("Training collaborative model...", flush=True)
    model = SVD(n_factors=100, n_epochs=20, random_state=42).fit(trainset)
    popularity = item_popularity(trainset)

    held_out = defaultdict(list)
    for user_id, item_id, _ in testset:
        if user_id in trainset._raw2inner_id_users:
            held_out[user_id].append(item_id)

    baseline_hits, baseline_ndcgs = [], []
    hybrid_hits, hybrid_ndcgs = [], []
    for user_id, items_for_user in held_out.items():
        baseline = [item for item in popularity if item not in {
            trainset.to_raw_iid(item_id) for item_id, _ in trainset.ur[trainset.to_inner_uid(user_id)]
        }][:TOP_K]
        hybrid = recommend_hybrid(model, trainset, item_features, user_id, popularity)
        baseline_hit, baseline_ndcg = top_k_metrics(baseline, items_for_user)
        hybrid_hit, hybrid_ndcg = top_k_metrics(hybrid, items_for_user)
        baseline_hits.append(baseline_hit)
        baseline_ndcgs.append(baseline_ndcg)
        hybrid_hits.append(hybrid_hit)
        hybrid_ndcgs.append(hybrid_ndcg)

    full_trainset = data.build_full_trainset()
    full_model = SVD(n_factors=100, n_epochs=20, random_state=42).fit(full_trainset)
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": full_model, "trainset": full_trainset, "items": items, "item_features": item_features, "popularity": item_popularity(full_trainset)},
        ARTIFACT_PATH,
    )
    return {
        "project": "ScholarS hybrid course recommendation prototype",
        "content_stand_in": "course category/level/skills represented by substitute genre features",
        "users": full_trainset.n_users,
        "items": full_trainset.n_items,
        "baseline_hit_rate_at_5": float(np.mean(baseline_hits)),
        "baseline_ndcg_at_5": float(np.mean(baseline_ndcgs)),
        "hybrid_hit_rate_at_5": float(np.mean(hybrid_hits)),
        "hybrid_ndcg_at_5": float(np.mean(hybrid_ndcgs)),
        "artifact": str(ARTIFACT_PATH),
    }


if __name__ == "__main__":
    for key, value in evaluate().items():
        print(f"{key}: {value}")
