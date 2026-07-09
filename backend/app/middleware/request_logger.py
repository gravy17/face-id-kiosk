"""
app/middleware/request_logger.py
Runs on every request:
  1. Generates a UUID request ID
  2. Sets request_id and client_ip on context vars (picked up by logger)
  3. Logs request start and completion with timing
  4. Attaches X-Request-ID to the response header
"""
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import client_ip_var, get_logger, request_id_var

logger = get_logger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: any) -> Response:
        request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
        client_ip  = request.client.host if request.client else "unknown"

        # Set context vars — structlog picks these up automatically
        token_rid = request_id_var.set(request_id)
        token_ip  = client_ip_var.set(client_ip)

        start = time.perf_counter()

        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            raise
        finally:
            request_id_var.reset(token_rid)
            client_ip_var.reset(token_ip)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
