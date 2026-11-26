"""FastAPI application factory, auth endpoints, SPA serving, and /api/* 404 isolation."""

from __future__ import annotations

import contextlib
import gzip
import io
import os
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config import ConfigManager
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

_GZIP_BYPASS_CT_PREFIXES: tuple[str, ...] = (
    "text/event-stream",
    "video/",
    "audio/",
    "image/",
    "application/zip",
)


class SSESafeGZipMiddleware:
    """ASGI GZip middleware that never buffers SSE or already-compressed media streams."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        minimum_size: int = 1024,
        compresslevel: int = 6,
    ) -> None:
        self.app = app
        self.minimum_size = max(0, int(minimum_size))
        self.compresslevel = int(compresslevel)

    @staticmethod
    def _is_sse_path(path: str) -> bool:
        norm = path.rstrip("/")
        return norm == "/api/events" or norm.endswith("/events")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        headers = Headers(scope=scope)
        accept_encoding = headers.get("accept-encoding", "")
        accept = headers.get("accept", "")
        if (
            "gzip" not in accept_encoding.lower()
            or self._is_sse_path(path)
            or "text/event-stream" in accept.lower()
        ):
            await self.app(scope, receive, send)
            return

        responder = _SSESafeGZipResponder(
            self.app,
            minimum_size=self.minimum_size,
            compresslevel=self.compresslevel,
        )
        await responder(scope, receive, send)


class _SSESafeGZipResponder:
    def __init__(
        self,
        app: ASGIApp,
        *,
        minimum_size: int,
        compresslevel: int,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel
        self.initial_message: Message | None = None
        self.started = False
        self.passthrough = False
        self.gzip_buffer: io.BytesIO | None = None
        self.gzip_file: gzip.GzipFile | None = None
        self.send: Send | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.send = send
        await self.app(scope, receive, self.send_with_gzip)

    @staticmethod
    def _should_bypass_response(message: Message) -> bool:
        status_code = int(message.get("status", 200))
        if status_code < 200 or status_code in (204, 206, 304):
            return True
        resp_headers = Headers(raw=message.get("headers", []))
        if "content-encoding" in resp_headers:
            return True
        ct = resp_headers.get("content-type", "").strip().lower()
        return any(ct.startswith(prefix) for prefix in _GZIP_BYPASS_CT_PREFIXES)

    async def send_with_gzip(self, message: Message) -> None:
        assert self.send is not None
        msg_type = message["type"]
        if msg_type == "http.response.start":
            if self._should_bypass_response(message):
                self.passthrough = True
                self.started = True
                await self.send(message)
            else:
                self.initial_message = message
            return

        if msg_type != "http.response.body":
            await self.send(message)
            return

        if self.passthrough:
            await self.send(message)
            return

        body = message.get("body", b"")
        more_body = bool(message.get("more_body", False))

        if not self.started:
            assert self.initial_message is not None
            self.started = True
            if not more_body:
                if len(body) < self.minimum_size:
                    self.passthrough = True
                    await self.send(self.initial_message)
                    await self.send(message)
                    return
                compressed = gzip.compress(body, compresslevel=self.compresslevel)
                mut_headers = MutableHeaders(raw=self.initial_message["headers"])
                mut_headers["Content-Encoding"] = "gzip"
                mut_headers["Content-Length"] = str(len(compressed))
                mut_headers.add_vary_header("Accept-Encoding")
                await self.send(self.initial_message)
                await self.send(
                    {
                        "type": "http.response.body",
                        "body": compressed,
                        "more_body": False,
                    }
                )
                return

            # Streaming compressible response
            mut_headers = MutableHeaders(raw=self.initial_message["headers"])
            mut_headers["Content-Encoding"] = "gzip"
            mut_headers.add_vary_header("Accept-Encoding")
            if "content-length" in mut_headers:
                del mut_headers["content-length"]
            self.gzip_buffer = io.BytesIO()
            self.gzip_file = gzip.GzipFile(
                mode="wb",
                fileobj=self.gzip_buffer,
                compresslevel=self.compresslevel,
            )
            await self.send(self.initial_message)

        assert self.gzip_file is not None and self.gzip_buffer is not None
        if body:
            self.gzip_file.write(body)
        if not more_body:
            self.gzip_file.close()
        else:
            self.gzip_file.flush()
        chunk = self.gzip_buffer.getvalue()
        self.gzip_buffer.seek(0)
        self.gzip_buffer.truncate()
        await self.send(
            {
                "type": "http.response.body",
                "body": chunk,
                "more_body": more_body,
            }
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
    # Validate config eagerly on startup so misconfigured directories fail fast.
    cfg_init = ConfigManager(config_dir=cfg_dir, workspace_dir=ws_path)
    raw_max_gb = (
        cfg_init.settings.get("io", {})
        .get("preview_cache", {})
        .get("max_cache_gb", 10.0)
    )
    max_cache_gb = float(raw_max_gb) if raw_max_gb is not None else None
    preview_cache = PreviewCache(ws_path, max_workers=2, max_cache_gb=max_cache_gb)
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

    app.add_middleware(SSESafeGZipMiddleware, minimum_size=1024)

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
