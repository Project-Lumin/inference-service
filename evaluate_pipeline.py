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

from service import VideoPayload, recommend_top_videos


@dataclass(frozen=True)
class RatingEvent:
    user_id: str
    movie_id: str
    rating: float
    timestamp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline end-to-end ranking evaluation for the full recommendation pipeline on MovieLens10M."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to MovieLens10M directory containing ratings.dat/movies.dat")
    parser.add_argument("--model-path", type=Path, default=Path("optimized_svdpp.pkl"), help="Path to pickled Surprise SVD/SVD++ model")
    parser.add_argument("--k", type=int, default=20, help="Final top-N for ranking metrics")
    parser.add_argument("--top-k-candidates", type=int, default=50, help="Pipeline prefilter size passed to recommend_top_videos")
    parser.add_argument("--candidate-size", type=int, default=200, help="Total candidate pool size per user request")
    parser.add_argument("--min-interactions", type=int, default=20, help="Minimum interactions per user before splitting")
    parser.add_argument("--test-positives", type=int, default=3, help="Number of most recent positive interactions held out per user")
    parser.add_argument("--watched-size", type=int, default=5, help="How many recent watched videos to provide as context")
    parser.add_argument("--relevance-threshold", type=float, default=4.0, help="Rating threshold for relevant/positive items")
    parser.add_argument("--max-users", type=int, default=0, help="Limit evaluated users for smoke runs (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def _split_line(raw: str, maxsplit: int) -> list[str]:
    return raw.rstrip("\n").split("::", maxsplit=maxsplit)


def load_movies(movies_path: Path) -> dict[str, VideoPayload]:
    movies: dict[str, VideoPayload] = {}
    with movies_path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = _split_line(line, maxsplit=2)
            if len(parts) != 3:
                continue
            movie_id, title, genres = parts
            movies[movie_id] = VideoPayload(id=movie_id, name=title, genre=genres)
    return movies


def load_tags(tags_path: Path) -> dict[str, str]:
    movie_tags: dict[str, list[str]] = defaultdict(list)
    if not tags_path.exists():
        return {}

    with tags_path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = _split_line(line, maxsplit=3)
            if len(parts) != 4:
                continue
            _, movie_id, tag, _ = parts
            tag = tag.strip()
            if tag:
                movie_tags[movie_id].append(tag)

    return {movie_id: "|".join(sorted(set(tags))) for movie_id, tags in movie_tags.items()}


def load_ratings(ratings_path: Path) -> list[RatingEvent]:
    ratings: list[RatingEvent] = []
    with ratings_path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = _split_line(line, maxsplit=3)
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, timestamp = parts
            ratings.append(
                RatingEvent(
                    user_id=user_id,
                    movie_id=movie_id,
                    rating=float(rating),
                    timestamp=int(timestamp),
                )
            )
    return ratings


def enrich_movie_catalog(base_movies: dict[str, VideoPayload], tags_by_movie: dict[str, str]) -> dict[str, VideoPayload]:
    enriched: dict[str, VideoPayload] = {}
    for movie_id, payload in base_movies.items():
        enriched[movie_id] = VideoPayload(
            id=payload.id,
            name=payload.name,
            genre=payload.genre,
            tags=tags_by_movie.get(movie_id),
        )
    return enriched


def dcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            score += 1.0 / math.log2(rank + 1)
    return score


def idcg_at_k(num_relevant: int, k: int) -> float:
    upto = min(num_relevant, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, upto + 1))


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_user_histories(events: list[RatingEvent]) -> dict[str, list[RatingEvent]]:
    histories: dict[str, list[RatingEvent]] = defaultdict(list)
    for event in events:
        histories[event.user_id].append(event)
    for user_id in histories:
        histories[user_id].sort(key=lambda entry: entry.timestamp)
    return histories


def choose_watched_context(
    train_events: list[RatingEvent],
    movies: dict[str, VideoPayload],
    watched_size: int,
    relevance_threshold: float,
) -> list[VideoPayload]:
    positives = [entry for entry in train_events if entry.rating >= relevance_threshold]
    source = positives if len(positives) >= watched_size else train_events
    selected = source[-watched_size:]
    return [movies[event.movie_id] for event in selected if event.movie_id in movies]


def compute_stage_candidate_recall(
    model: object,
    user_id: str,
    candidate_ids: list[str],
    relevant_ids: set[str],
    top_k_candidates: int,
) -> float:
    if not relevant_ids:
        return 0.0

    scored: list[tuple[str, float]] = []
    for movie_id in candidate_ids:
        prediction = model.predict(user_id, movie_id)
        estimated = float(getattr(prediction, "est", prediction))
        scored.append((movie_id, estimated))

    scored.sort(key=lambda item: item[1], reverse=True)
    prefiltered_ids = {movie_id for movie_id, _ in scored[:top_k_candidates]}
    hits = len(prefiltered_ids & relevant_ids)
    return hits / len(relevant_ids)


def run_evaluation(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    ratings_path = args.data_dir / "ratings.dat"
    movies_path = args.data_dir / "movies.dat"
    tags_path = args.data_dir / "tags.dat"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError("MovieLens10M files not found. Expected ratings.dat and movies.dat in --data-dir")

    print("Loading model and dataset...")
    model = load_model(args.model_path)
    raw_movies = load_movies(movies_path)
    tags_by_movie = load_tags(tags_path)
    movies = enrich_movie_catalog(raw_movies, tags_by_movie)
    events = load_ratings(ratings_path)

    histories = build_user_histories(events)
    all_movie_ids = set(movies.keys())

    metric_ndcg: list[float] = []
    metric_recall: list[float] = []
    metric_hit_rate: list[float] = []
    metric_mrr: list[float] = []
    metric_stage_recall: list[float] = []

    failure_counts = defaultdict(int)
    evaluated_users = 0
    recommended_items: set[str] = set()

    user_ids = list(histories.keys())
    random.shuffle(user_ids)

    for user_index, user_id in enumerate(user_ids, start=1):
        if args.max_users > 0 and evaluated_users >= args.max_users:
            break

        user_events = histories[user_id]
        if len(user_events) < args.min_interactions:
            failure_counts["too_few_interactions"] += 1
            continue

        positive_events = [entry for entry in user_events if entry.rating >= args.relevance_threshold]
        if len(positive_events) <= args.test_positives:
            failure_counts["too_few_positive_events"] += 1
            continue

        test_positive_events = positive_events[-args.test_positives :]
        test_positive_ids = {entry.movie_id for entry in test_positive_events if entry.movie_id in all_movie_ids}
        if not test_positive_ids:
            failure_counts["no_valid_test_positives"] += 1
            continue

        cutoff_timestamp = min(entry.timestamp for entry in test_positive_events)
        train_events = [entry for entry in user_events if entry.timestamp < cutoff_timestamp]
        if not train_events:
            failure_counts["empty_train_after_split"] += 1
            continue

        watched_videos = choose_watched_context(
            train_events=train_events,
            movies=movies,
            watched_size=args.watched_size,
            relevance_threshold=args.relevance_threshold,
        )
        if not watched_videos:
            failure_counts["empty_watched_context"] += 1
            continue

        seen_train_ids = {entry.movie_id for entry in train_events}
        unseen_pool = list(all_movie_ids - seen_train_ids)
        unseen_pool = [movie_id for movie_id in unseen_pool if movie_id not in test_positive_ids]

        neg_needed = max(args.candidate_size - len(test_positive_ids), 0)
        if len(unseen_pool) < neg_needed:
            negatives = unseen_pool
        else:
            negatives = random.sample(unseen_pool, k=neg_needed)

        candidate_ids = list(test_positive_ids) + negatives
        if len(candidate_ids) < len(test_positive_ids):
            failure_counts["candidate_construction_failed"] += 1
            continue

        random.shuffle(candidate_ids)
        candidate_videos = [movies[movie_id] for movie_id in candidate_ids if movie_id in movies]
        if not candidate_videos:
            failure_counts["empty_candidate_videos"] += 1
            continue

        try:
            stage_recall = compute_stage_candidate_recall(
                model=model,
                user_id=user_id,
                candidate_ids=[video.id for video in candidate_videos],
                relevant_ids=test_positive_ids,
                top_k_candidates=args.top_k_candidates,
            )
            metric_stage_recall.append(stage_recall)

            with redirect_stdout(io.StringIO()):
                recommendations = recommend_top_videos(
                    svd_model=model,
                    user_id=user_id,
                    candidate_videos=candidate_videos,
                    watched_videos=watched_videos,
                    top_k_candidates=args.top_k_candidates,
                    top_n_final=args.k,
                )
        except Exception:
            failure_counts["pipeline_exception"] += 1
            continue

        if not recommendations:
            failure_counts["empty_recommendations"] += 1
            continue

        topk = recommendations[: args.k]
        recommended_items.update(topk)
        hits = [item_id for item_id in topk if item_id in test_positive_ids]

        recall = len(hits) / len(test_positive_ids)
        hit_rate = 1.0 if hits else 0.0

        reciprocal_rank = 0.0
        for rank, item_id in enumerate(topk, start=1):
            if item_id in test_positive_ids:
                reciprocal_rank = 1.0 / rank
                break

        dcg = dcg_at_k(topk, test_positive_ids, args.k)
        idcg = idcg_at_k(len(test_positive_ids), args.k)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        metric_recall.append(recall)
        metric_hit_rate.append(hit_rate)
        metric_mrr.append(reciprocal_rank)
        metric_ndcg.append(ndcg)

        evaluated_users += 1

        if evaluated_users % 200 == 0:
            print(f"Progress: evaluated_users={evaluated_users} (seen {user_index} users)")

    print("\n===== Full Pipeline Evaluation (MovieLens10M) =====")
    print(f"Evaluated users         : {evaluated_users}")
    print(f"k                       : {args.k}")
    print(f"top_k_candidates        : {args.top_k_candidates}")
    print(f"candidate_size          : {args.candidate_size}")
    print(f"relevance_threshold     : {args.relevance_threshold}")
    print("\nPrimary metric")
    print(f"nDCG@{args.k:<2}              : {safe_mean(metric_ndcg):.6f}")
    print("\nSecondary metrics")
    print(f"Recall@{args.k:<2}            : {safe_mean(metric_recall):.6f}")
    print(f"HitRate@{args.k:<2}           : {safe_mean(metric_hit_rate):.6f}")
    print(f"MRR@{args.k:<2}               : {safe_mean(metric_mrr):.6f}")
    print(f"Candidate Recall@{args.top_k_candidates:<2}: {safe_mean(metric_stage_recall):.6f}")

    # Placeholder explicit output for catalog coverage until top-k collection is persisted.
    coverage = (len(recommended_items) / len(all_movie_ids)) if all_movie_ids else 0.0
    print(f"Catalog coverage       : {coverage:.6f}")

    print("\nFailure counts")
    if not failure_counts:
        print("none")
    else:
        for key in sorted(failure_counts):
            print(f"{key:28s}: {failure_counts[key]}")


if __name__ == "__main__":
    parsed_args = parse_args()
    run_evaluation(parsed_args)
