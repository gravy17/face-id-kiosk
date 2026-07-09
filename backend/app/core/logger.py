"""
app/core/logger.py
Structured logging using structlog. Every log entry carries:
  - timestamp, level, logger name
  - request_id (injected by middleware via contextvars)
  - any extra fields passed at the call site
"""
import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ── Request context var ────────────────────────────────────────────────────
# Set by RequestLoggerMiddleware at the start of each request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
client_ip_var:  ContextVar[str] = ContextVar("client_ip",  default="-")


def _add_request_context(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor — injects request_id and client_ip from context."""
    event_dict["request_id"] = request_id_var.get("-")
    event_dict["client_ip"]  = client_ip_var.get("-")
    return event_dict


def configure_logging(debug: bool = False) -> None:
    """
    Call once at app startup (main.py).
    Outputs JSON in production, coloured console in debug mode.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_request_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if debug:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Quieten noisy libraries
    for noisy in ("uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger. Usage: logger = get_logger(__name__)"""
    return structlog.get_logger(name)
