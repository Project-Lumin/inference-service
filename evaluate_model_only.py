from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd


MODEL_PATH = Path("optimized_svdpp.pkl")
TESTSET_CSV = Path(r"C:/Users/Ridma Premaratne/Downloads/testset.csv")
RELEVANCE_THRESHOLD = 3.5
K_VALUES = (5, 10, 20)


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def load_testset(testset_csv: Path) -> list[tuple[str, str, float]]:
    if not testset_csv.exists():
        raise FileNotFoundError(f"Testset file not found: {testset_csv}")

    df = pd.read_csv(testset_csv)
    expected = {"userId", "movieId", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"testset csv missing columns: {sorted(missing)}")

    df = df.dropna(subset=["userId", "movieId", "rating"]).copy()
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()

    return [(row.userId, row.movieId, float(row.rating)) for row in df.itertuples(index=False)]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def precision_at_k(ranked_pairs: list[tuple[float, float]], k: int, threshold: float) -> float:
    topk = ranked_pairs[:k]
    if not topk:
        return 0.0
    hits = sum(1 for _, true_rating in topk if true_rating >= threshold)
    return hits / k


def average_precision_at_k(ranked_pairs: list[tuple[float, float]], k: int, threshold: float) -> float:
    topk = ranked_pairs[:k]
    relevant_count = sum(1 for _, true_rating in topk if true_rating >= threshold)
    if relevant_count == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, (_, true_rating) in enumerate(topk, start=1):
        if true_rating >= threshold:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / relevant_count


def ndcg_at_k(ranked_pairs: list[tuple[float, float]], k: int, threshold: float) -> float:
    topk = ranked_pairs[:k]
    dcg = 0.0
    for rank, (_, true_rating) in enumerate(topk, start=1):
        if true_rating >= threshold:
            dcg += 1.0 / math.log2(rank + 1)

    relevant_count = sum(1 for _, true_rating in topk if true_rating >= threshold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(relevant_count))
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(ranked_pairs: list[tuple[float, float]], threshold: float) -> float:
    for rank, (_, true_rating) in enumerate(ranked_pairs, start=1):
        if true_rating >= threshold:
            return 1.0 / rank
    return 0.0


def evaluate_model(model: object, testset: list[tuple[str, str, float]]) -> None:
    predictions = model.test(testset)

    user_predictions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pred in predictions:
        user_predictions[pred.uid].append((float(pred.est), float(pred.r_ui)))

    for uid in user_predictions:
        user_predictions[uid].sort(key=lambda pair: pair[0], reverse=True)

    print("===== Model-Only Evaluation =====")
    print(f"test_rows   : {len(testset)}")
    print(f"users       : {len(user_predictions)}")
    print(f"threshold   : {RELEVANCE_THRESHOLD}")

    for k in K_VALUES:
        precisions: list[float] = []
        maps: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []

        for ranked_pairs in user_predictions.values():
            precisions.append(precision_at_k(ranked_pairs, k, RELEVANCE_THRESHOLD))
            maps.append(average_precision_at_k(ranked_pairs, k, RELEVANCE_THRESHOLD))
            ndcgs.append(ndcg_at_k(ranked_pairs, k, RELEVANCE_THRESHOLD))
            mrrs.append(mrr_at_k(ranked_pairs[:k], RELEVANCE_THRESHOLD))

        print(f"\n@K={k}")
        print(f"MRR@{k:<2}       : {safe_mean(mrrs):.6f}")
        print(f"MAP@{k:<2}       : {safe_mean(maps):.6f}")
        print(f"NDCG@{k:<2}      : {safe_mean(ndcgs):.6f}")
        print(f"Precision@{k:<2}  : {safe_mean(precisions):.6f}")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    testset = load_testset(TESTSET_CSV)
    evaluate_model(model, testset)
