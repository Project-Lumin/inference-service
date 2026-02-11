"""Ranking utilities for SVD-based recommendations with content reranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VideoPayload:
    id: str
    file_id: str | None = None
    name: str | None = None
    genre: str | None = None
    length: str | None = None
    tags: str | None = None
    date_posted: str | None = None
    metadata: dict | None = None


def _tokenize_metadata(video: VideoPayload) -> set[str]:
    """Build a simple bag-of-words from title + genre + tags."""
    parts = []
    if video.name:
        parts.append(video.name)
    if video.genre:
        parts.append(video.genre.replace("|", " "))
    if video.tags:
        parts.append(video.tags.replace("|", " "))

    combined = " ".join(parts)
    return {tok for tok in combined.lower().split() if tok}


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def recommend_top_videos(
    svd_model: object,
    user_id: str,
    candidate_videos: Sequence[VideoPayload],
    watched_videos: Sequence[VideoPayload],
    *,
    top_k_candidates: int = 50,
    top_n_final: int = 20,
) -> list[str]:
    """
    Rank candidate videos for a user using SVD predictions, then rerank by
    simple metadata similarity to watched videos.

    Args:
        svd_model: A fitted scikit-surprise SVD-like model exposing .predict(uid, iid).
        user_id: Target user identifier.
        candidate_videos: Videos to score (e.g., 100 items with metadata).
        watched_videos: Recently watched video items (e.g., 5 items).
        top_k_candidates: How many highest SVD scores to keep before reranking.
        top_n_final: How many videos to return after similarity reranking.

    Returns:
        List of top_n_final video ids.
    """

    if top_n_final <= 0:
        raise ValueError("top_n_final must be positive")
    if top_k_candidates <= 0:
        raise ValueError("top_k_candidates must be positive")
    if not candidate_videos:
        raise ValueError("candidate_videos is empty")

    # 1) Score all candidates with the SVD model
    scored = []
    for video in candidate_videos:
        pred = svd_model.predict(user_id, video.id)
        score = float(getattr(pred, "est", pred))
        scored.append((video, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    for video, score in scored:
        print(f"[recommend] {video.id}: {score}")
    prefiltered = [vid for vid, _ in scored[:top_k_candidates]]
    score_lookup = {video.id: score for video, score in scored}

    if not watched_videos:
        return [v.id for v in prefiltered[:top_n_final]]

    watched_tokens = [_tokenize_metadata(v) for v in watched_videos]

    def video_similarity(video: VideoPayload) -> float:
        tokens = _tokenize_metadata(video)
        if not tokens:
            return 0.0
        sims = [_jaccard_similarity(tokens, wt) for wt in watched_tokens if wt]
        return sum(sims) / len(sims) if sims else 0.0

    reranked = sorted(prefiltered, key=video_similarity, reverse=True)
    top_final = reranked[:top_n_final]
    top_final_sorted = sorted(top_final, key=lambda v: score_lookup.get(v.id, 0.0), reverse=True)
    return [v.id for v in top_final_sorted]
