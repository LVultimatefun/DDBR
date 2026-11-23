"""Claim lease management and 3-layer human review endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from ...models.review import ReviewWritePayload
from ...store import (
    AuthorizationError,
    ClaimConflictError,
    GtFrozenError,
    InvalidTransitionError,
    NotFoundError,
    ReadOnlyFieldError,
    RepositoryError,
    WatchCoverageError,
)
from ..auth import AuthContext, get_app_state, require_auth
from ..state import AppState

router = APIRouter(prefix="/api/episodes", tags=["reviews"])


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str | None = None


def _resolve_claim_user(
    body: ClaimRequest | None,
    auth: AuthContext,
    request: Request | None = None,
) -> str:
    explicit_x_user = (
        (request.headers.get("X-User") or "").strip() if request is not None else ""
    )
    if body is not None and body.user is not None:
        u = body.user.strip()
        if not u:
            raise HTTPException(status_code=422, detail="user must be non-empty")
        if auth.role != "lead" and explicit_x_user and u != auth.user:
            raise HTTPException(
                status_code=409,
                detail=f"Reviewer {auth.user!r} cannot act on claim for {u!r}",
            )
        return u
    return auth.user


def _map_repo_error(exc: RepositoryError) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AuthorizationError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, (ClaimConflictError, GtFrozenError, InvalidTransitionError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (WatchCoverageError, ReadOnlyFieldError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.post("/{ep_id:path}/claim")
def claim_episode(
    ep_id: str,
    request: Request,
    body: ClaimRequest | None = None,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    user = _resolve_claim_user(body, auth, request)
    cfg = state.get_config()
    ttl_s = float(cfg.settings.get("review", {}).get("claim_ttl_s", 120.0))
    state.repo.reap_expired_claims()
    try:
        res = state.repo.claim_episode(ep_id, user, ttl_s=ttl_s)
    except RepositoryError as exc:
        raise _map_repo_error(exc) from exc
    state.sse.publish({"type": "episode_claimed", "episode_id": ep_id, "claimed_by": user})
    return res


@router.post("/{ep_id:path}/renew")
def renew_claim(
    ep_id: str,
    request: Request,
    body: ClaimRequest | None = None,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    user = _resolve_claim_user(body, auth, request)
    cfg = state.get_config()
    ttl_s = float(cfg.settings.get("review", {}).get("claim_ttl_s", 120.0))
    state.repo.reap_expired_claims()
    try:
        return state.repo.renew_claim(ep_id, user, ttl_s=ttl_s)
    except RepositoryError as exc:
        raise _map_repo_error(exc) from exc


@router.post("/{ep_id:path}/release")
def release_claim(
    ep_id: str,
    request: Request,
    body: ClaimRequest | None = None,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    user = _resolve_claim_user(body, auth, request)
    state.repo.reap_expired_claims()
    try:
        next_status = state.repo.release_claim(ep_id, user)
    except RepositoryError as exc:
        raise _map_repo_error(exc) from exc
    state.sse.publish({"type": "episode_released", "episode_id": ep_id, "status": next_status})
    return {"episode_id": ep_id, "status": next_status}


@router.get("/{ep_id:path}/reviews")
def list_episode_reviews(
    ep_id: str,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        items = state.repo.list_reviews(ep_id)
    except RepositoryError as exc:
        raise _map_repo_error(exc) from exc
    return {
        "items": [r.model_dump(mode="json") for r in items],
        "total": len(items),
    }


@router.post("/{ep_id:path}/reviews")
def submit_episode_review(
    ep_id: str,
    payload: ReviewWritePayload,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    cfg = state.get_config()
    min_watch = float(cfg.settings.get("review", {}).get("min_watch_coverage", 0.95))
    state.repo.reap_expired_claims()
    try:
        artifact = state.repo.save_review(
            ep_id,
            author=auth.user,
            role=auth.role,
            payload=payload,
            min_watch_coverage=min_watch,
        )
    except RepositoryError as exc:
        raise _map_repo_error(exc) from exc
    state.sse.publish(
        {
            "type": "review_saved",
            "episode_id": ep_id,
            "rev_id": artifact.rev_id,
            "action": artifact.action,
            "author": artifact.author,
        }
    )
    return artifact.model_dump(mode="json")
