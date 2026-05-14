from __future__ import annotations

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


MODEL_PATH = Path("optimized_svdpp.pkl")
RATINGS_CSV = Path(r"C:/Users/Ridma Premaratne/Downloads/rating.csv")
TESTSET_CSV = Path(r"C:/Users/Ridma Premaratne/Downloads/test_set.csv")
MOVIES_CSV: Path | None = Path(r"C:/Users/Ridma Premaratne/Downloads/movie.csv/movie.csv")
TAGS_CSV: Path | None = Path(r"C:/Users/Ridma Premaratne/Downloads/movie.csv/tag.csv/tag.csv")
RELEVANCE_THRESHOLD = 3.5
K_VALUES = (5, 10, 20)
TOP_K_CANDIDATES = 50
CANDIDATE_SIZE = 200
WATCHED_SIZE = 5
MIN_INTERACTIONS = 10
SEED = 42


@dataclass(frozen=True)
class RatingEvent:
    user_id: str
    movie_id: str
    rating: float
    timestamp: int


@dataclass(frozen=True)
class EvalCase:
    user_id: str
    relevant_ids: set[str]
    candidate_ids: list[str]
    watched_videos: list[VideoPayload]


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def load_ratings(ratings_csv: Path) -> pd.DataFrame:
    if not ratings_csv.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_csv}")

    df = pd.read_csv(ratings_csv)
    expected = {"userId", "movieId", "rating", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ratings csv missing columns: {sorted(missing)}")

    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_int"] = (df["timestamp"].astype("int64") // 10**9).astype(int)
    return df


def load_testset(testset_csv: Path) -> pd.DataFrame:
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

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def load_tags(tags_csv: Path | None) -> dict[str, str]:
    if not tags_csv or not tags_csv.exists():
        return {}

    df_tags = pd.read_csv(tags_csv)
    expected = {"movieId", "tag"}
    missing = expected - set(df_tags.columns)
    if missing:
        raise ValueError(f"tags csv missing columns: {sorted(missing)}")

    df_tags = df_tags.dropna(subset=["movieId", "tag"]).copy()
    df_tags["movieId"] = df_tags["movieId"].astype(str)
    df_tags["tag"] = df_tags["tag"].astype(str)

    grouped = df_tags.groupby("movieId")["tag"].apply(lambda series: "|".join(sorted(set(series.tolist()))))
    return grouped.to_dict()


def load_movie_catalog(
    df_ratings: pd.DataFrame,
    testset_df: pd.DataFrame,
    movies_csv: Path | None,
    tags_csv: Path | None,
) -> dict[str, VideoPayload]:
    tags_by_movie = load_tags(tags_csv)

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
            catalog[movie_id] = VideoPayload(
                id=movie_id,
                name=str(getattr(row, "title", None)) if getattr(row, "title", None) is not None else None,
                genre=str(getattr(row, "genres", None)) if getattr(row, "genres", None) is not None else None,
                tags=tags_by_movie.get(movie_id),
            )
        return catalog

    movie_ids = sorted(set(df_ratings["movieId"].astype(str).unique()) | set(testset_df["movieId"].astype(str).unique()))
    return {movie_id: VideoPayload(id=movie_id, tags=tags_by_movie.get(movie_id)) for movie_id in movie_ids}


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


def choose_watched_context(train_events: list[RatingEvent], movies: dict[str, VideoPayload]) -> list[VideoPayload]:
    positives = [event for event in train_events if event.rating >= RELEVANCE_THRESHOLD]
    source = positives if len(positives) >= WATCHED_SIZE else train_events
    selected = source[-WATCHED_SIZE:]
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


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def summarize_metrics(name: str, per_user_rankings: list[list[str]], per_user_relevant: list[set[str]]) -> None:
    print(f"\n{name}")
    for k in K_VALUES:
        precisions = []
        maps = []
        ndcgs = []
        mrrs = []
        for ranked_ids, relevant_ids in zip(per_user_rankings, per_user_relevant):
            precisions.append(precision_at_k(ranked_ids, relevant_ids, k))
            maps.append(average_precision_at_k(ranked_ids, relevant_ids, k))
            ndcgs.append(ndcg_at_k(ranked_ids, relevant_ids, k))
            mrrs.append(mrr_at_k(ranked_ids, relevant_ids, k))

        print(f"@K={k}")
        print(f"MRR@{k:<2}       : {safe_mean(mrrs):.6f}")
        print(f"MAP@{k:<2}       : {safe_mean(maps):.6f}")
        print(f"NDCG@{k:<2}      : {safe_mean(ndcgs):.6f}")
        print(f"Precision@{k:<2}  : {safe_mean(precisions):.6f}")


def build_eval_cases(train_df: pd.DataFrame, testset_df: pd.DataFrame, catalog: dict[str, VideoPayload]) -> tuple[list[EvalCase], dict[str, int]]:
    histories = build_user_histories(train_df)
    test_groups = {uid: group.copy() for uid, group in testset_df.groupby("userId")}
    all_item_ids = set(catalog.keys())
    skipped = defaultdict(int)
    cases: list[EvalCase] = []

    for user_id, eval_events in test_groups.items():
        user_history = histories.get(user_id, [])
        if len(user_history) < MIN_INTERACTIONS:
            skipped["too_few_interactions"] += 1
            continue

        relevant_ids = {
            str(movie_id)
            for movie_id, rating in zip(eval_events["movieId"], eval_events["rating"])
            if float(rating) >= RELEVANCE_THRESHOLD and str(movie_id) in all_item_ids
        }
        if not relevant_ids:
            skipped["no_relevant_eval_items"] += 1
            continue

        watched_videos = choose_watched_context(user_history, catalog)
        if not watched_videos:
            skipped["empty_watched_context"] += 1
            continue

        seen_train_ids = {event.movie_id for event in user_history}
        unseen_pool = list(all_item_ids - seen_train_ids - relevant_ids)
        neg_needed = max(CANDIDATE_SIZE - len(relevant_ids), 0)
        negatives = random.sample(unseen_pool, k=min(len(unseen_pool), neg_needed))
        candidate_ids = list(relevant_ids) + negatives
        random.shuffle(candidate_ids)
        if not candidate_ids:
            skipped["empty_candidates"] += 1
            continue

        cases.append(
            EvalCase(
                user_id=user_id,
                relevant_ids=relevant_ids,
                candidate_ids=candidate_ids,
                watched_videos=watched_videos,
            )
        )

    return cases, dict(skipped)


def evaluate_svd_ranking(model: object, eval_cases: list[EvalCase]) -> None:
    per_user_rankings: list[list[str]] = []
    per_user_relevant: list[set[str]] = []

    for case in eval_cases:
        scored = []
        for item_id in case.candidate_ids:
            pred = model.predict(case.user_id, item_id)
            est = float(getattr(pred, "est", pred))
            scored.append((item_id, est))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked_ids = [item_id for item_id, _ in scored[: max(K_VALUES)]]

        per_user_rankings.append(ranked_ids)
        per_user_relevant.append(case.relevant_ids)

    print("===== SVD-Only Ranking (Same Candidates) =====")
    print(f"evaluated_users : {len(eval_cases)}")
    summarize_metrics("SVD-only metrics", per_user_rankings, per_user_relevant)


def evaluate_pipeline(model: object, eval_cases: list[EvalCase], catalog: dict[str, VideoPayload]) -> None:

    per_user_rankings: list[list[str]] = []
    per_user_relevant: list[set[str]] = []
    skipped = defaultdict(int)

    for case in eval_cases:
        candidate_videos = [catalog[item_id] for item_id in case.candidate_ids if item_id in catalog]
        if not candidate_videos:
            skipped["empty_candidates"] += 1
            continue

        try:
            with redirect_stdout(io.StringIO()):
                recs = recommend_top_videos(
                    svd_model=model,
                    user_id=case.user_id,
                    candidate_videos=candidate_videos,
                    watched_videos=case.watched_videos,
                    top_k_candidates=TOP_K_CANDIDATES,
                    top_n_final=max(K_VALUES),
                )
        except Exception:
            skipped["pipeline_exception"] += 1
            continue

        if not recs:
            skipped["empty_recommendations"] += 1
            continue

        per_user_rankings.append(recs)
        per_user_relevant.append(case.relevant_ids)

    print("\n===== Ranking Pipeline Evaluation =====")
    print(f"evaluated_users : {len(per_user_rankings)}")
    print(f"skipped_users   : {sum(skipped.values())}")
    summarize_metrics("Pipeline metrics", per_user_rankings, per_user_relevant)

    print("\nSkipped counts")
    if not skipped:
        print("none")
    else:
        for key in sorted(skipped):
            print(f"{key:24s}: {skipped[key]}")


if __name__ == "__main__":
    random.seed(SEED)
    model = load_model(MODEL_PATH)
    train_df = load_ratings(RATINGS_CSV)
    testset_df = load_testset(TESTSET_CSV)
    catalog = load_movie_catalog(train_df, testset_df, MOVIES_CSV, TAGS_CSV)
    eval_cases, prefilter_skips = build_eval_cases(train_df, testset_df, catalog)

    metadata_items = sum(1 for video in catalog.values() if video.name or video.genre or video.tags)

    print("===== Shared Evaluation Set =====")
    print(f"test_rows       : {len(testset_df)}")
    print(f"evaluated_users : {len(eval_cases)}")
    print(f"prefilter_skips : {sum(prefilter_skips.values())}")
    print(f"metadata_items  : {metadata_items}/{len(catalog)}")
    if prefilter_skips:
        for key in sorted(prefilter_skips):
            print(f"{key:24s}: {prefilter_skips[key]}")

    evaluate_svd_ranking(model, eval_cases)
    evaluate_pipeline(model, eval_cases, catalog)
