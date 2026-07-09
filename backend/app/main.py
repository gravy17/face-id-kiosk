"""
app/main.py
FastAPI application factory.
Registers middleware, routers, rate limiting, CORS, and lifecycle hooks.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import factsheet, register, verify
from app.core.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.database.db import close_db, init_db
from app.middleware.request_logger import RequestLoggerMiddleware

settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger   = get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    logger.info(
        "Starting up",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    await init_db()
    _start_cleanup_task(app)

    yield

    logger.info("Shutting down")
    await close_db()


def _start_cleanup_task(app: FastAPI) -> None:
    """Schedule the background cleanup task."""
    import asyncio
    from app.database.db import AsyncSessionFactory
    from app.database.repository import get_repository

    async def _cleanup_loop() -> None:
        interval = settings.CLEANUP_INTERVAL_HOURS * 3600
        while True:
            await asyncio.sleep(interval)
            try:
                async with AsyncSessionFactory() as session:
                    repo    = get_repository(session)
                    deleted = await repo.delete_records_older_than(settings.CLEANUP_AGE_DAYS)
                    await session.commit()
                    if deleted:
                        logger.info("Background cleanup ran", deleted=deleted)
            except Exception as exc:
                logger.error("Background cleanup failed", error=str(exc))

    asyncio.create_task(_cleanup_loop())


# ── App factory ───────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── State ─────────────────────────────────────────────────────────────
    app.state.limiter = limiter

    # ── Middleware (order matters — outermost first) ───────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestLoggerMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Rate limits on specific routes ───────────────────────────────────
    @app.middleware("http")
    async def apply_rate_limits(request: Request, call_next):
        return await call_next(request)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(register.router)
    app.include_router(verify.router)
    app.include_router(factsheet.router)

    # Apply per-route rate limiting via decorators in each router,
    # or via slowapi dependencies added here:
    limiter.limit(settings.RATE_LIMIT_REGISTER)(
        app.routes[next(i for i, r in enumerate(app.routes) if getattr(r, "path", "") == "/register")]
    )

    return app


app = create_app()


# ── Dev entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
