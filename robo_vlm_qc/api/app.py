"""FastAPI application factory, auth endpoints, SPA serving, and /api/* 404 isolation."""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..io import PreviewCache
from ..store import Repository
from .auth import AuthContext, get_app_state, require_auth
from .routers import episodes, exports, ingest, jobs, lab, reviews, settings
from .state import (
    AppState,
    AuthConfig,
    resolve_default_config_dir,
    resolve_default_web_dist,
)

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)


def _resolve_cors_origins(cors_origins: Sequence[str] | None) -> list[str]:
    if cors_origins is not None:
        return [str(o).strip() for o in cors_origins if str(o).strip()]
    env_raw = os.environ.get("ROBO_VLM_QC_CORS_ORIGINS", "").strip()
    if env_raw:
        parsed = [part.strip() for part in env_raw.split(",") if part.strip()]
        if parsed:
            return parsed
    return list(DEFAULT_CORS_ORIGINS)


def _resolve_tokens(
    *,
    reviewer_token: str | None,
    lead_token: str | None,
    tokens: dict[str, str] | None,
    no_auth: bool,
) -> AuthConfig:
    rev = reviewer_token
    lead = lead_token
    if tokens:
        if rev is None and "reviewer" in tokens:
            rev = tokens["reviewer"]
        if lead is None and "lead" in tokens:
            lead = tokens["lead"]
    if rev is None:
        rev = os.environ.get("ROBO_VLM_QC_REVIEWER_TOKEN") or None
    if lead is None:
        lead = os.environ.get("ROBO_VLM_QC_LEAD_TOKEN") or None
    if not no_auth and not rev and not lead:
        raise ValueError(
            "Authentication tokens required when no_auth=False: provide "
            "reviewer_token/lead_token (or ROBO_VLM_QC_REVIEWER_TOKEN / "
            "ROBO_VLM_QC_LEAD_TOKEN), or explicitly pass no_auth=True."
        )
    return AuthConfig(reviewer_token=rev, lead_token=lead, no_auth=bool(no_auth))


def create_app(
    workspace: Path | str = Path("./workspace"),
    config_dir: Path | None = None,
    reviewer_token: str | None = None,
    lead_token: str | None = None,
    no_auth: bool = False,
    tokens: dict[str, str] | None = None,
    web_dist_dir: Path | None = None,
    allowed_data_roots: Sequence[Path | str] | None = None,
    cors_origins: Sequence[str] | None = None,
    recover_stale_jobs: bool = False,
) -> FastAPI:
    ws_path = Path(workspace).resolve()
    ws_path.mkdir(parents=True, exist_ok=True)
    cfg_dir = resolve_default_config_dir(config_dir)

    repo = Repository(ws_path)
    preview_cache = PreviewCache(ws_path, max_workers=2)
    auth_cfg = _resolve_tokens(
        reviewer_token=reviewer_token,
        lead_token=lead_token,
        tokens=tokens,
        no_auth=no_auth,
    )
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="robo-vlm-qc-job")
    default_web_dist = resolve_default_web_dist(web_dist_dir)
    norm_allowed_roots = (
        [Path(r).expanduser().resolve() for r in allowed_data_roots]
        if allowed_data_roots is not None
        else None
    )
    app_state = AppState(
        workspace=ws_path,
        config_dir=cfg_dir,
        repo=repo,
        preview_cache=preview_cache,
        auth=auth_cfg,
        executor=executor,
        web_dist=default_web_dist,
        allowed_data_roots=norm_allowed_roots,
    )

    # Validate config eagerly on startup so misconfigured directories fail fast.
    app_state.get_config()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if recover_stale_jobs:
            with contextlib.suppress(Exception):
                repo.recover_stale_active_jobs()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                app_state.cancel_all_jobs()
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False)
            with contextlib.suppress(Exception):
                repo.close()

    app = FastAPI(
        title="robo_vlm_qc Studio API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.ctx = app_state

    origins = _resolve_cors_origins(cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=("*" not in origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/auth/me")
    def auth_me(
        auth: AuthContext = Depends(require_auth),
        state: AppState = Depends(get_app_state),
    ) -> dict[str, Any]:
        return {"role": auth.role, "user": auth.user, "no_auth": state.auth.no_auth}

    @app.get("/api/labels")
    def shared_labels(
        auth: AuthContext = Depends(require_auth),
        state: AppState = Depends(get_app_state),
    ) -> dict[str, Any]:
        _ = auth
        return state.get_config().labels.model_dump(mode="json")

    # Register suffix review routes BEFORE bare /api/episodes/{ep_id:path}
    app.include_router(reviews.router)
    app.include_router(episodes.router)
    app.include_router(jobs.router)
    app.include_router(exports.router)
    app.include_router(lab.router)
    app.include_router(settings.router)
    app.include_router(ingest.router)

    assets_dir = default_web_dist / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="spa-assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Any:
        norm = full_path.lstrip("/")
        if norm == "api" or norm.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"API route not found: /{norm}")

        dist_root = app_state.web_dist
        if norm:
            candidate = (dist_root / norm).resolve()
            try:
                is_inside = candidate.is_relative_to(dist_root.resolve())
            except ValueError:
                is_inside = False
            if is_inside and candidate.is_file():
                return FileResponse(candidate)

        index_file = dist_root / "index.html"
        if index_file.is_file():
            return FileResponse(index_file, media_type="text/html")
        raise HTTPException(status_code=404, detail="Web SPA dist/index.html not built")

    return app


build_app = create_app
