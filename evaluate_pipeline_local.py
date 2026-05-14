from __future__ import annotations

import argparse
import io
import math
import pickle
import random
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from service import VideoPayload, recommend_top_videos


RATINGS_CSV = Path(r"C:/Users/Ridma Premaratne/Downloads/rating.csv")
TESTSET_CSV = Path(r"C:/Users/Ridma Premaratne/Downloads/testset.csv")


@dataclass(frozen=True)
class RatingEvent:
    user_id: str
    movie_id: str
    rating: float
    timestamp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full recommendation pipeline from local CSVs using notebook-style preprocessing and an explicit test set."
    )
    parser.add_argument("--movies-csv", type=Path, default=None, help="Optional movies metadata CSV (movieId,title,genres)")
    parser.add_argument("--model-path", type=Path, default=Path("optimized_svdpp.pkl"), help="Path to pickled Surprise model")
    parser.add_argument("--k", type=int, default=20, help="Top-N for ranking metrics")
    parser.add_argument("--top-k-candidates", type=int, default=50, help="Prefilter size in pipeline")
    parser.add_argument("--candidate-size", type=int, default=200, help="Candidate pool size per user")
    parser.add_argument("--relevance-threshold", type=float, default=4.0, help="Rating threshold for relevant items")
    parser.add_argument("--recent-days", type=int, default=90, help="Notebook-style recent-day filter window")
    parser.add_argument("--min-interactions", type=int, default=10, help="Minimum training interactions per user after filtering")
    parser.add_argument("--watched-size", type=int, default=5, help="Watched context size for reranker")
    parser.add_argument("--max-users", type=int, default=0, help="Limit number of users for smoke test (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def load_and_filter_ratings(ratings_csv: Path, recent_days: int) -> tuple[pd.DataFrame, dict[str, int]]:
    if not ratings_csv.exists():
        raise FileNotFoundError(f"ratings csv not found: {ratings_csv}")

    df = pd.read_csv(ratings_csv)
    expected = {"userId", "movieId", "rating", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ratings csv missing columns: {sorted(missing)}")

    rows_total = len(df)
    null_counts = {key: int(value) for key, value in df.isnull().sum().to_dict().items()}
    duplicate_rows = int(df.duplicated().sum())

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "userId", "movieId", "rating"]).copy()

    cutoff = df["timestamp"].max() - pd.Timedelta(days=recent_days)
    df = df[df["timestamp"] >= cutoff].copy()

    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()

    df["ts_int"] = (df["timestamp"].astype("int64") // 10**9).astype(int)

    stats = {
        "rows_total": rows_total,
        "rows_after_filter": int(len(df)),
        "duplicate_rows": duplicate_rows,
        "users": int(df["userId"].nunique()),
        "items": int(df["movieId"].nunique()),
    }

    print("Null counts:")
    print(null_counts)
    print(f"Duplicate rows: {duplicate_rows}")

    return df, stats


def load_testset(testset_csv: Path) -> pd.DataFrame:
    if not testset_csv.exists():
        raise FileNotFoundError(f"testset csv not found: {testset_csv}")

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

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def load_movie_catalog(df_ratings: pd.DataFrame, movies_csv: Path | None) -> dict[str, VideoPayload]:
    if movies_csv and movies_csv.exists():
        df_movies = pd.read_csv(movies_csv)
        expected = {"movieId", "title", "genres"}
        missing = expected - set(df_movies.columns)
        if missing:
            raise ValueError(f"movies csv missing columns: {sorted(missing)}")

        df_movies = df_movies.dropna(subset=["movieId"]).copy()
        df_movies["movieId"] = df_movies["movieId"].astype(str)

        catalog: dict[str, VideoPayload] = {}
        for row in df_movies.itertuples(index=False):
            movie_id = str(getattr(row, "movieId"))
            title = getattr(row, "title", None)
            genres = getattr(row, "genres", None)
            catalog[movie_id] = VideoPayload(
                id=movie_id,
                name=str(title) if title is not None else None,
                genre=str(genres) if genres is not None else None,
            )
        return catalog

    movie_ids = sorted(df_ratings["movieId"].astype(str).unique().tolist())
    return {movie_id: VideoPayload(id=movie_id) for movie_id in movie_ids}


def build_user_histories(df: pd.DataFrame) -> dict[str, list[RatingEvent]]:
    histories: dict[str, list[RatingEvent]] = defaultdict(list)
    for row in df[["userId", "movieId", "rating", "ts_int"]].itertuples(index=False):
        histories[str(row.userId)].append(
            RatingEvent(
                user_id=str(row.userId),
                movie_id=str(row.movieId),
                rating=float(row.rating),
                timestamp=int(row.ts_int),
            )
        )
    for user_id in histories:
        histories[user_id].sort(key=lambda event: event.timestamp)
    return histories


def build_test_events(df: pd.DataFrame) -> dict[str, list[RatingEvent]]:
    histories: dict[str, list[RatingEvent]] = defaultdict(list)

    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["ts_int"] = (df["timestamp"].astype("int64") // 10**9).astype(int)
    else:
        df = df.copy()
        df["ts_int"] = range(len(df))

    for row in df[["userId", "movieId", "rating", "ts_int"]].itertuples(index=False):
        histories[str(row.userId)].append(
            RatingEvent(
                user_id=str(row.userId),
                movie_id=str(row.movieId),
                rating=float(row.rating),
                timestamp=int(row.ts_int),
            )
        )

    for user_id in histories:
        histories[user_id].sort(key=lambda event: event.timestamp)

    return histories


def choose_watched_context(
    train_events: list[RatingEvent],
    movies: dict[str, VideoPayload],
    watched_size: int,
    relevance_threshold: float,
) -> list[VideoPayload]:
    positives = [event for event in train_events if event.rating >= relevance_threshold]
    source = positives if len(positives) >= watched_size else train_events
    selected = source[-watched_size:]
    return [movies[event.movie_id] for event in selected if event.movie_id in movies]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for item_id in topk if item_id in relevant_ids)
    return hits / k


def average_precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / min(len(relevant_ids), k)


def dcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            score += 1.0 / math.log2(rank + 1)
    return score


def idcg_at_k(num_relevant: int, k: int) -> float:
    upto = min(num_relevant, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, upto + 1))


def run_evaluation(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    model = load_model(args.model_path)
    df_ratings, stats = load_and_filter_ratings(RATINGS_CSV, args.recent_days)
    df_testset = load_testset(TESTSET_CSV)
    catalog = load_movie_catalog(df_ratings, args.movies_csv)

    histories = build_user_histories(df_ratings)
    test_histories = build_test_events(df_testset)
    all_item_ids = set(catalog.keys())

    metric_ndcg: list[float] = []
    metric_mrr: list[float] = []
    metric_precision: list[float] = []
    metric_map: list[float] = []

    failures = defaultdict(int)
    evaluated_users = 0
    recommended_items: set[str] = set()

    user_ids = list(histories.keys())
    random.shuffle(user_ids)

    for index, user_id in enumerate(user_ids, start=1):
        if args.max_users > 0 and evaluated_users >= args.max_users:
            break

        events = histories[user_id]
        if len(events) < args.min_interactions:
            failures["too_few_interactions"] += 1
            continue

        eval_events = test_histories.get(user_id, [])
        relevant_ids = {event.movie_id for event in eval_events if event.rating >= args.relevance_threshold and event.movie_id in all_item_ids}
        if not relevant_ids:
            failures["no_relevant_eval_items"] += 1
            continue

        train_events = events

        watched_videos = choose_watched_context(
            train_events=train_events,
            movies=catalog,
            watched_size=args.watched_size,
            relevance_threshold=args.relevance_threshold,
        )
        if not watched_videos:
            failures["empty_watched_context"] += 1
            continue

        seen_train_ids = {event.movie_id for event in train_events}
        unseen_pool = list(all_item_ids - seen_train_ids - relevant_ids)

        neg_needed = max(args.candidate_size - len(relevant_ids), 0)
        negatives = random.sample(unseen_pool, k=min(len(unseen_pool), neg_needed))

        candidate_ids = list(relevant_ids) + negatives
        if not candidate_ids:
            failures["empty_candidates"] += 1
            continue

        random.shuffle(candidate_ids)
        candidate_videos = [catalog[item_id] for item_id in candidate_ids if item_id in catalog]

        try:
            with redirect_stdout(io.StringIO()):
                recs = recommend_top_videos(
                    svd_model=model,
                    user_id=user_id,
                    candidate_videos=candidate_videos,
                    watched_videos=watched_videos,
                    top_k_candidates=args.top_k_candidates,
                    top_n_final=args.k,
                )
        except Exception:
            failures["pipeline_exception"] += 1
            continue

        if not recs:
            failures["empty_recommendations"] += 1
            continue

        topk = recs[: args.k]
        recommended_items.update(topk)

        hits = [item_id for item_id in topk if item_id in relevant_ids]

        reciprocal_rank = 0.0
        for rank, item_id in enumerate(topk, start=1):
            if item_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        dcg = dcg_at_k(topk, relevant_ids, args.k)
        idcg = idcg_at_k(len(relevant_ids), args.k)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        precision = precision_at_k(topk, relevant_ids, args.k)
        ap = average_precision_at_k(topk, relevant_ids, args.k)

        metric_mrr.append(reciprocal_rank)
        metric_ndcg.append(ndcg)
        metric_precision.append(precision)
        metric_map.append(ap)

        evaluated_users += 1

        if evaluated_users % 200 == 0:
            print(f"Progress: evaluated_users={evaluated_users} (seen {index} users)")

    coverage = (len(recommended_items) / len(all_item_ids)) if all_item_ids else 0.0

    print("\n===== Local CSV Full Pipeline Evaluation =====")
    print(f"rows_total                 : {stats['rows_total']}")
    print(f"rows_after_recent_filter   : {stats['rows_after_filter']}")
    print(f"unique_users               : {stats['users']}")
    print(f"unique_items               : {stats['items']}")
    print(f"test_rows                  : {len(df_testset)}")
    print(f"evaluated_users            : {evaluated_users}")
    print(f"k                          : {args.k}")
    print(f"top_k_candidates           : {args.top_k_candidates}")

    print("\nPrimary metric")
    print(f"nDCG@{args.k:<2}                  : {safe_mean(metric_ndcg):.6f}")

    print("\nSecondary metrics")
    print(f"MRR@{args.k:<2}                   : {safe_mean(metric_mrr):.6f}")
    print(f"MAP@{args.k:<2}                   : {safe_mean(metric_map):.6f}")
    print(f"Precision@{args.k:<2}             : {safe_mean(metric_precision):.6f}")
    print(f"Catalog coverage           : {coverage:.6f}")

    print("\nFailure counts")
    if not failures:
        print("none")
    else:
        for key in sorted(failures):
            print(f"{key:28s}: {failures[key]}")


if __name__ == "__main__":
    run_evaluation(parse_args())
