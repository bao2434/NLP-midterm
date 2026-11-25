"""HTTP client for the Kim Han-Nom Sentence Segmentation API (https://kimhannom.clc.hcmus.edu.vn/).

Follows the specification in ``TaiLieuAPI_HanNom_CLC.docx``:
- Base domain: https://kimhannom.clc.hcmus.edu.vn/ (segmentation & punctuation endpoints)
- Endpoints:
  - POST /api/web/clc-sinonom/separate-sentences (punctuation + sentence segmentation)
  - POST /api/web/clc-sinonom/sentence-punctuation (punctuation only)
- Auth header: ``Authorization: Bearer <token>`` (website cookie ``token``)
- User-Agent header: MUST NOT start with ``python-requests``
- Rate limits: 25/min, 50/hour, 200/day, 600/week
- Congestion rule: insert a delay between requests and avoid concurrent bursts
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .utils import split_cjk_sentences

DEFAULT_BASE_URL = "https://kimhannom.clc.hcmus.edu.vn"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

VALID_GENRES = (
    "administrative",
    "buddhism",
    "history",
    "literature",
    "medical",
    "philosophy",
    "poetry",
)
VALID_PUNCTUATION_GROUPS = (
    "all",
    "fullstop",
    "other_punctuation",
)
VALID_ENDPOINTS = (
    "separate-sentences",
    "sentence-punctuation",
)


class APIError(RuntimeError):
    """Raised when the Kim Han-Nom API returns a non-success response."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        payload: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.payload = payload


class RateLimitError(APIError):
    """Raised when a local rate-limit guard or HTTP 429 is triggered."""


@dataclass
class SegmentationAPIConfig:
    """Configuration for calling the Kim Han-Nom Sentence Segmentation API."""

    # Endpoint & authentication
    base_url: str = DEFAULT_BASE_URL
    token: Optional[str] = None
    user_agent: str = DEFAULT_USER_AGENT

    # Segmentation API parameters
    endpoint: str = "separate-sentences"  # "separate-sentences" or "sentence-punctuation"
    genre: str = "medical"                # default: "medical" for traditional medicine documents
    punctuation_group: str = "all"        # "all", "fullstop", or "other_punctuation"

    # Rate limiting & retries (API rules: 25/min, 50/hr, 200/day, 600/week)
    rate_limit_mode: str = "limited"      # "limited" (enforce limits) or "unlimited" (call continuously without limits)
    request_delay: Optional[float] = None  # seconds between API calls (default: 0.0 in unlimited mode, 3.0 in limited mode)
    max_requests_per_minute: Optional[int] = 25
    max_requests_per_hour: Optional[int] = 50
    max_requests_per_day: Optional[int] = 200
    max_requests_per_week: Optional[int] = 600
    max_requests_per_run: Optional[int] = None
    max_consecutive_errors: int = 5       # stop early on repeated API errors in limited mode (0 disables; ignored in unlimited mode)
    respect_rate_limits: Optional[bool] = None  # backward-compatible alias for rate_limit_mode
    timeout: float = 300.0
    max_retries: int = 3
    retry_backoff: float = 5.0

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("HANNOM_API_TOKEN") or None
        if self.token is not None:
            cleaned_token = self.token.strip()
            if cleaned_token.lower().startswith("bearer "):
                cleaned_token = cleaned_token[7:].strip()
            self.token = cleaned_token or None
        if self.endpoint not in VALID_ENDPOINTS:
            raise ValueError(f"endpoint must be one of {VALID_ENDPOINTS}")
        if self.genre not in VALID_GENRES:
            raise ValueError(f"genre must be one of {VALID_GENRES}")
        if self.punctuation_group not in VALID_PUNCTUATION_GROUPS:
            raise ValueError(
                f"punctuation_group must be one of {VALID_PUNCTUATION_GROUPS}"
            )
        if self.rate_limit_mode not in {"limited", "unlimited"}:
            raise ValueError("rate_limit_mode must be 'limited' or 'unlimited'")
        if self.respect_rate_limits is None:
            self.respect_rate_limits = self.rate_limit_mode == "limited"
        else:
            if not self.respect_rate_limits:
                self.rate_limit_mode = "unlimited"
            elif self.rate_limit_mode == "unlimited":
                raise ValueError(
                    "Conflicting settings: rate_limit_mode='unlimited' cannot be combined "
                    "with respect_rate_limits=True."
                )
        if self.request_delay is None:
            self.request_delay = 0.0 if self.is_unlimited else 3.0
        if self.request_delay < 0:
            raise ValueError("request_delay must be >= 0")
        for field_name in (
            "max_requests_per_minute",
            "max_requests_per_hour",
            "max_requests_per_day",
            "max_requests_per_week",
            "max_requests_per_run",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0 or None")
        if self.user_agent.strip().lower().startswith("python-requests"):
            raise ValueError(
                "User-Agent header must NOT start with 'python-requests' "
                "(per TaiLieuAPI_HanNom_CLC specification)."
            )
        self.base_url = self.base_url.rstrip("/")

    @property
    def is_unlimited(self) -> bool:
        """Return True when local API call limits are disabled."""
        return self.rate_limit_mode == "unlimited"

    def rate_limit_summary(self) -> str:
        """Return a human-readable summary of the active API call limit settings."""
        if self.is_unlimited:
            return "unlimited"
        parts: list[str] = []
        if self.max_requests_per_minute:
            parts.append(f"{self.max_requests_per_minute}/min")
        if self.max_requests_per_hour:
            parts.append(f"{self.max_requests_per_hour}/hr")
        if self.max_requests_per_day:
            parts.append(f"{self.max_requests_per_day}/day")
        if self.max_requests_per_week:
            parts.append(f"{self.max_requests_per_week}/wk")
        if self.max_requests_per_run:
            parts.append(f"{self.max_requests_per_run}/run")
        return f"limited ({', '.join(parts)})" if parts else "limited (no caps set)"


# Alias for consistency with ocr_api.APIConfig
APIConfig = SegmentationAPIConfig


class HanNomSegmentationClient:
    """Sequential, rate-limit-aware client for the Kim Han-Nom sentence segmentation API."""

    SEPARATE_SENTENCES_PATH = "/api/web/clc-sinonom/separate-sentences"
    SENTENCE_PUNCTUATION_PATH = "/api/web/clc-sinonom/sentence-punctuation"

    def __init__(self, config: Optional[SegmentationAPIConfig] = None):
        self.config = config or SegmentationAPIConfig()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Content-Type": "application/json",
            }
        )
        if self.config.token:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.config.token}"}
            )

        self._last_request_ts: float = 0.0
        self._minute_window: deque[float] = deque()
        self._hour_window: deque[float] = deque()
        self._day_window: deque[float] = deque()
        self._week_window: deque[float] = deque()
        self.request_count: int = 0

    def _prune_windows(self, now: float) -> None:
        while self._minute_window and now - self._minute_window[0] >= 60.0:
            self._minute_window.popleft()
        while self._hour_window and now - self._hour_window[0] >= 3600.0:
            self._hour_window.popleft()
        while self._day_window and now - self._day_window[0] >= 86400.0:
            self._day_window.popleft()
        while self._week_window and now - self._week_window[0] >= 604800.0:
            self._week_window.popleft()

    def _wait_for_slot(self) -> None:
        """Enforce inter-request delay and (when in limited mode) request rate limits."""
        cfg = self.config
        limited = not cfg.is_unlimited

        if limited and cfg.max_requests_per_run and self.request_count >= cfg.max_requests_per_run:
            raise RateLimitError(
                f"Reached max_requests_per_run ({cfg.max_requests_per_run})."
            )

        now = time.monotonic()
        if self._last_request_ts > 0 and cfg.request_delay > 0:
            elapsed = now - self._last_request_ts
            if elapsed < cfg.request_delay:
                time.sleep(cfg.request_delay - elapsed)
                now = time.monotonic()

        if limited:
            self._prune_windows(now)

            if cfg.max_requests_per_week and len(self._week_window) >= cfg.max_requests_per_week:
                raise RateLimitError(
                    f"Reached weekly API rate limit ({cfg.max_requests_per_week} requests/week)."
                )
            if cfg.max_requests_per_day and len(self._day_window) >= cfg.max_requests_per_day:
                raise RateLimitError(
                    f"Reached daily API rate limit ({cfg.max_requests_per_day} requests/day)."
                )
            if cfg.max_requests_per_hour and len(self._hour_window) >= cfg.max_requests_per_hour:
                raise RateLimitError(
                    f"Reached hourly API rate limit ({cfg.max_requests_per_hour} requests/hour). "
                    "Resume later to continue from the last completed chunk."
                )
            if cfg.max_requests_per_minute and len(self._minute_window) >= cfg.max_requests_per_minute:
                sleep_for = 60.0 - (now - self._minute_window[0]) + 0.5
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                self._prune_windows(now)

            self._minute_window.append(now)
            self._hour_window.append(now)
            self._day_window.append(now)
            self._week_window.append(now)

        self._last_request_ts = now
        self.request_count += 1

    def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an HTTP request, handle retries, and validate the API response envelope."""
        url = f"{self.config.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            self._wait_for_slot()
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.config.timeout,
                    **kwargs,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff * attempt)
                    continue
                raise APIError(f"Network error calling {path}: {exc}") from exc

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if attempt < self.config.max_retries:
                    if retry_after and retry_after.isdigit():
                        wait_s = min(float(retry_after), 65.0)
                        time.sleep(wait_s)
                        continue
                    if self.config.is_unlimited:
                        time.sleep(self.config.retry_backoff * attempt)
                        continue
                raise RateLimitError(
                    f"HTTP 429 Too Many Requests from {path}",
                    status_code=429,
                )

            if response.status_code >= 500:
                last_error = APIError(
                    f"HTTP {response.status_code} from {path}: {response.text[:300]}",
                    status_code=response.status_code,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff * attempt)
                    continue
                raise last_error

            try:
                payload = response.json()
            except ValueError as exc:
                raise APIError(
                    f"Non-JSON response (HTTP {response.status_code}) from {path}: "
                    f"{response.text[:300]}",
                    status_code=response.status_code,
                ) from exc

            if response.status_code >= 400:
                raise APIError(
                    f"HTTP {response.status_code} from {path}: {payload}",
                    code=str(payload.get("code")) if isinstance(payload, dict) else None,
                    status_code=response.status_code,
                    payload=payload if isinstance(payload, dict) else None,
                )

            if not isinstance(payload, dict):
                raise APIError(f"Unexpected JSON payload from {path}: {payload!r}")

            code = str(payload.get("code", ""))
            is_success = bool(payload.get("is_success", False))
            if not is_success or code != "000000":
                msg = payload.get("message")
                if isinstance(msg, dict):
                    detail = " - ".join(
                        str(msg[k]) for k in ("title", "text", "debug_msg") if msg.get(k)
                    )
                else:
                    detail = str(msg)
                raise APIError(
                    f"API error on {path} (code={code}): {detail or payload}",
                    code=code,
                    status_code=response.status_code,
                    payload=payload,
                )

            return payload.get("data") or {}

        raise APIError(f"Failed request to {path}: {last_error}")

    def separate_sentences(
        self,
        text: str,
        *,
        genre: Optional[str] = None,
        punctuation_group: Optional[str] = None,
    ) -> dict[str, Any]:
        """Punctuate and segment text via POST /api/web/clc-sinonom/separate-sentences."""
        cfg = self.config
        req_genre = cfg.genre if genre is None else genre
        req_punct = cfg.punctuation_group if punctuation_group is None else punctuation_group

        if req_genre not in VALID_GENRES:
            raise ValueError(f"genre must be one of {VALID_GENRES}")
        if req_punct not in VALID_PUNCTUATION_GROUPS:
            raise ValueError(f"punctuation_group must be one of {VALID_PUNCTUATION_GROUPS}")

        data = self._request_json(
            "POST",
            self.SEPARATE_SENTENCES_PATH,
            json={
                "text": text,
                "genre": req_genre,
                "punctuation_group": req_punct,
            },
        )

        raw_sentences = data.get("sentences")
        if isinstance(raw_sentences, list):
            sentences = [str(s).strip() for s in raw_sentences if str(s).strip()]
            segmented_text = "\n".join(sentences)
        else:
            segmented_text = str(raw_sentences or data.get("result") or "").strip()
            sentences = [
                line.strip()
                for line in segmented_text.splitlines()
                if line.strip()
            ]

        return {
            "text": segmented_text,
            "sentences": sentences,
            "endpoint": "separate-sentences",
            "genre": req_genre,
            "punctuation_group": req_punct,
        }

    def punctuate_sentences(
        self,
        text: str,
        *,
        genre: Optional[str] = None,
        punctuation_group: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add punctuation via POST /api/web/clc-sinonom/sentence-punctuation."""
        cfg = self.config
        req_genre = cfg.genre if genre is None else genre
        req_punct = cfg.punctuation_group if punctuation_group is None else punctuation_group

        if req_genre not in VALID_GENRES:
            raise ValueError(f"genre must be one of {VALID_GENRES}")
        if req_punct not in VALID_PUNCTUATION_GROUPS:
            raise ValueError(f"punctuation_group must be one of {VALID_PUNCTUATION_GROUPS}")

        data = self._request_json(
            "POST",
            self.SENTENCE_PUNCTUATION_PATH,
            json={
                "text": text,
                "genre": req_genre,
                "punctuation_group": req_punct,
            },
        )

        punctuated = str(data.get("result") or "").strip()
        sentences = split_cjk_sentences(punctuated) if punctuated else []
        return {
            "text": "\n".join(sentences) if sentences else punctuated,
            "sentences": sentences,
            "labels": data.get("labels") or [],
            "endpoint": "sentence-punctuation",
            "genre": req_genre,
            "punctuation_group": req_punct,
        }

    def segment_text(
        self,
        text: str,
        *,
        genre: Optional[str] = None,
        punctuation_group: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Segment a text chunk using the configured API endpoint."""
        target = self.config.endpoint if endpoint is None else endpoint
        if target == "sentence-punctuation":
            return self.punctuate_sentences(
                text,
                genre=genre,
                punctuation_group=punctuation_group,
            )
        return self.separate_sentences(
            text,
            genre=genre,
            punctuation_group=punctuation_group,
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HanNomSegmentationClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
