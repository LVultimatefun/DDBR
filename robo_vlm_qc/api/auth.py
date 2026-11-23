"""Token authentication and Reviewer / Lead RBAC dependencies."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Query, Request

from ..models.review import ReviewRole
from .state import AppState


@dataclass(frozen=True)
class AuthContext:
    role: ReviewRole
    user: str
    token: str


def get_app_state(request: Request) -> AppState:
    return request.app.state.ctx


def require_auth(
    request: Request,
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
    token: str | None = Query(default=None),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> AuthContext:
    state = get_app_state(request)
    auth_cfg = state.auth
    raw_token = (x_auth_token if x_auth_token is not None else token) or ""
    raw_token = raw_token.strip()

    role: ReviewRole | None = None
    if auth_cfg.no_auth:
        if (
            raw_token
            and auth_cfg.reviewer_token
            and secrets.compare_digest(raw_token, auth_cfg.reviewer_token)
            and (
                not auth_cfg.lead_token
                or not secrets.compare_digest(raw_token, auth_cfg.lead_token)
            )
        ):
            role = "reviewer"
        else:
            role = "lead"
    else:
        if not raw_token:
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token (X-Auth-Token header or ?token= query)",
            )
        if auth_cfg.lead_token and secrets.compare_digest(raw_token, auth_cfg.lead_token):
            role = "lead"
        elif auth_cfg.reviewer_token and secrets.compare_digest(
            raw_token, auth_cfg.reviewer_token
        ):
            role = "reviewer"
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
            )

    user = (x_user or "").strip() or role
    return AuthContext(role=role, user=user, token=raw_token)


def require_lead(auth: AuthContext = Depends(require_auth)) -> AuthContext:
    if auth.role != "lead":
        raise HTTPException(
            status_code=403,
            detail="Lead role required for this endpoint",
        )
    return auth
