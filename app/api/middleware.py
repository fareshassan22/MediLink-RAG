import os
import time
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from threading import Lock


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_api_key: Optional[str] = None
_request_counts: Dict[str, list] = defaultdict(list)
_counts_lock = Lock()


def get_api_key() -> Optional[str]:
    """Get the configured API key from environment."""
    global _api_key
    if _api_key is None:
        _api_key = os.getenv("MEDILINK_API_KEY", "")
    return _api_key


async def verify_api_key(request: Request) -> bool:
    """Verify the API key from request headers.

    If MEDILINK_API_KEY is not set, authentication is skipped (dev mode).
    """
    api_key = get_api_key()
    if not api_key:
        # No key configured — open access (development)
        return True

    provided_key = request.headers.get("X-API-Key")
    return provided_key == api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 30):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        with _counts_lock:
            now = time.time()
            window_start = now - self.window_size

            if client_ip in _request_counts:
                _request_counts[client_ip] = [
                    ts for ts in _request_counts[client_ip] if ts > window_start
                ]

            if len(_request_counts[client_ip]) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                )

            _request_counts[client_ip].append(now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(_request_counts[client_ip])
        )
        return response


def get_rate_limit_status() -> Dict[str, int]:
    """Get current rate limit status for monitoring."""
    with _counts_lock:
        return {ip: len(times) for ip, times in _request_counts.items()}
