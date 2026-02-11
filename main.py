"""Minimal Flask service exposing SVD model predictions and recommendations."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

from service import VideoPayload, recommend_top_videos

MODEL_PATH = Path(__file__).resolve().parent / "svd_full_model.pkl"
PREFETCH_URL = "http://34.173.213.87:8000/v1/user/prefetched/videos"

app = Flask(__name__)


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with path.open("rb") as f:
        return pickle.load(f)


try:
    svd_model = load_model(MODEL_PATH)
except Exception as exc:  # pragma: no cover - load failure stops app
    # Fail fast on startup if the model is missing or broken
    raise RuntimeError(f"Could not load model: {exc}") from exc


def _notify_prefetched(user_id: str, videos: list[str]) -> tuple[Any, int | None]:
    payload = {"user_id": user_id, "videos": videos}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        PREFETCH_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_body = resp.read()
            try:
                body: Any = json.loads(raw_body) if raw_body else None
            except Exception:
                body = raw_body.decode("utf-8", errors="replace") if raw_body else None

            print(f"[recommend] prefetched notify status={resp.status}")
            return body, resp.status
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[recommend] prefetched notify failed: {exc}")
        return {"detail": f"prefetch notify failed: {exc}"}, None


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.post("/predict")
def predict() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    item_id = payload.get("item_id")

    if not user_id or not item_id:
        return {"detail": "user_id and item_id are required"}, 400

    try:
        prediction = svd_model.predict(str(user_id), str(item_id))
        estimate = float(getattr(prediction, "est", prediction))
        return {"user_id": str(user_id), "item_id": str(item_id), "estimate": estimate}, 200
    except Exception as exc:
        return {"detail": f"Prediction failed: {exc}"}, 500


def _parse_video_list(raw_list: Any, field_name: str) -> list[VideoPayload]:
    if not isinstance(raw_list, list):
        raise ValueError(f"{field_name} must be a list of objects")

    videos: list[VideoPayload] = []
    for entry in raw_list:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ValueError(f"Each item in {field_name} must include an 'id'")
        videos.append(VideoPayload(**entry))
    return videos


@app.post("/recommend")
def recommend() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    print(f"[recommend] request body: {payload}")
    user_id = payload.get("user_id")
    if not user_id:
        return {"detail": "user_id is required"}, 400

    try:
        candidate_videos = _parse_video_list(payload.get("video_ids"), "video_ids")
        watched_videos = _parse_video_list(payload.get("watched_video_ids"), "watched_video_ids")
    except ValueError as exc:
        return {"detail": str(exc)}, 400

    try:
        video_ids = recommend_top_videos(
            svd_model=svd_model,
            user_id=str(user_id),
            candidate_videos=candidate_videos,
            watched_videos=watched_videos,
            top_k_candidates=50,
            top_n_final=20,
        )
    except ValueError as exc:
        return {"detail": str(exc)}, 400
    except Exception as exc:
        return {"detail": f"Recommendation failed: {exc}"}, 500

    notify_body, notify_status = _notify_prefetched(str(user_id), video_ids)

    if notify_status is None:
        return notify_body, 502

    return notify_body, notify_status
