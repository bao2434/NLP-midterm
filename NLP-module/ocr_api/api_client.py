"""HTTP client for the Kim Han-Nom OCR API (https://kimhannom.fit.hcmus.edu.vn/).

Follows the specification in ``TaiLieuAPI_HanNom_CLC .docx``:
- Base domain: https://kimhannom.fit.hcmus.edu.vn/
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

from .utils import image_to_bytes, image_to_data_uri

DEFAULT_BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
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
class APIConfig:
    """Configuration for calling the Kim Han-Nom OCR API."""

    # Endpoint & authentication
    base_url: str = DEFAULT_BASE_URL
    token: Optional[str] = None
    user_agent: str = DEFAULT_USER_AGENT

    # OCR mode: "hannom" (image-upload + image-ocr) or "viet" (viet-ocr)
    mode: str = "hannom"

    # Han-Nom /image-ocr parameters
    ocr_id: int = 1                # 1: normal vertical, 2: admin, 3: scene, 4: normal horizontal
    lang_type: int = 1             # 0: unknown, 1: Han, 2: Nom
    reading_direction: int = 1     # 1: vertical (default), 2: horizontal
    epitaph: int = 0               # 0: normal document, 1: epitaph (van bia)
    tag: Optional[str] = None      # specialized model_id if assigned
    auto_classify: bool = False    # call /structure-classification before /image-ocr

    # Rate limiting & retries (API rules: 25/min, 50/hr, 200/day, 600/week)
    rate_limit_mode: str = "limited"  # "limited" (enforce limits) or "unlimited" (allow unlimited calls)
    request_delay: float = 3.0        # seconds between API calls to avoid congestion (0 disables)
    max_requests_per_minute: Optional[int] = 25
    max_requests_per_hour: Optional[int] = 50
    max_requests_per_day: Optional[int] = 200
    max_requests_per_week: Optional[int] = 600
    max_requests_per_run: Optional[int] = None
    max_consecutive_errors: int = 5   # stop early on repeated API errors to save quota (0 disables)
    respect_rate_limits: Optional[bool] = None  # backward-compatible alias for rate_limit_mode
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 5.0

    # PDF rendering & image encoding
    dpi: int = 200
    max_side: int = 1600
    image_format: str = "JPEG"     # "JPEG" or "PNG"
    jpeg_quality: int = 92

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("HANNOM_API_TOKEN") or None
        if self.mode not in {"hannom", "viet"}:
            raise ValueError("mode must be 'hannom' or 'viet'")
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
        if self.ocr_id not in {1, 2, 3, 4}:
            raise ValueError("ocr_id must be 1, 2, 3, or 4")
        if self.lang_type not in {0, 1, 2}:
            raise ValueError("lang_type must be 0, 1, or 2")
        if self.reading_direction not in {1, 2}:
            raise ValueError("reading_direction must be 1 or 2")
        if self.epitaph not in {0, 1}:
            raise ValueError("epitaph must be 0 or 1")
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


class HanNomAPIClient:
    """Sequential, rate-limit-aware client for the Kim Han-Nom web API."""

    UPLOAD_PATH = "/api/web/clc-sinonom/image-upload"
    CLASSIFY_PATH = "/api/web/clc-sinonom/structure-classification"
    OCR_PATH = "/api/web/clc-sinonom/image-ocr"
    VIET_OCR_PATH = "/api/web/clc-sinonom/viet-ocr"

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
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
                    "Resume later to continue from the last completed page."
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
                if retry_after and retry_after.isdigit() and attempt < self.config.max_retries:
                    wait_s = min(float(retry_after), 65.0)
                    time.sleep(wait_s)
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

    def upload_image(
        self,
        image_bytes: bytes,
        filename: str = "page.jpg",
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload an image via POST /api/web/clc-sinonom/image-upload and return file_name."""
        data = self._request_json(
            "POST",
            self.UPLOAD_PATH,
            files={"image_file": (filename, image_bytes, content_type)},
        )
        file_name = str(data.get("file_name", "")).strip()
        if not file_name:
            raise APIError(f"image-upload did not return file_name: {data}")
        return file_name

    def classify_image(
        self,
        file_name: str,
        *,
        ocr_id: int = 0,
        lang_type: int = 0,
        reading_direction: int = 0,
    ) -> dict[str, Any]:
        """Classify an uploaded image via POST /api/web/clc-sinonom/structure-classification."""
        return self._request_json(
            "POST",
            self.CLASSIFY_PATH,
            json={
                "file_name": file_name,
                "ocr_id": ocr_id,
                "lang_type": lang_type,
                "reading_direction": reading_direction,
            },
        )

    def ocr_image(
        self,
        file_name: str,
        *,
        ocr_id: Optional[int] = None,
        lang_type: Optional[int] = None,
        reading_direction: Optional[int] = None,
        epitaph: Optional[int] = None,
        tag: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run Han-Nom OCR via POST /api/web/clc-sinonom/image-ocr."""
        cfg = self.config
        body: dict[str, Any] = {
            "ocr_id": cfg.ocr_id if ocr_id is None else ocr_id,
            "lang_type": cfg.lang_type if lang_type is None else lang_type,
            "reading_direction": (
                cfg.reading_direction if reading_direction is None else reading_direction
            ),
            "epitaph": cfg.epitaph if epitaph is None else epitaph,
            "tag": cfg.tag if tag is None else tag,
            "file_name": file_name,
        }
        return self._request_json("POST", self.OCR_PATH, json=body)

    def viet_ocr_image(self, image_data_uri: str, file_name: str = "page.jpg") -> dict[str, Any]:
        """Run Vietnamese OCR via POST /api/web/clc-sinonom/viet-ocr."""
        return self._request_json(
            "POST",
            self.VIET_OCR_PATH,
            json={
                "file_name": file_name,
                "image": image_data_uri,
            },
        )

    def ocr_pil_image(self, image, filename: str = "page.jpg") -> dict[str, Any]:
        """OCR a PIL Image using the configured mode ('hannom' or 'viet')."""
        cfg = self.config
        fmt = cfg.image_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"
        ext = "jpg" if fmt == "JPEG" else fmt.lower()
        content_type = "image/jpeg" if fmt == "JPEG" else f"image/{ext}"
        if not filename.lower().endswith(f".{ext}"):
            filename = f"{filename.rsplit('.', 1)[0]}.{ext}"

        if cfg.mode == "viet":
            data_uri = image_to_data_uri(image, fmt=fmt, quality=cfg.jpeg_quality)
            data = self.viet_ocr_image(data_uri, file_name=filename)
            results = data.get("results") or []
            lines = [
                str(item.get("text", "")).strip()
                for item in results
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            full_text = str(data.get("full_text") or "\n".join(lines)).strip()
            return {
                "text": full_text,
                "lines": lines or ([full_text] if full_text else []),
                "avg_confidence": data.get("avg_confidence"),
                "detections": data.get("detections"),
                "results": results,
            }

        # Default: Han-Nom OCR (upload -> optional structure-classification -> image-ocr)
        raw_bytes = image_to_bytes(image, fmt=fmt, quality=cfg.jpeg_quality)
        uploaded_name = self.upload_image(
            raw_bytes, filename=filename, content_type=content_type
        )

        ocr_id = cfg.ocr_id
        lang_type = cfg.lang_type
        reading_direction = cfg.reading_direction
        epitaph = cfg.epitaph
        classification: Optional[dict[str, Any]] = None
        data: Optional[dict[str, Any]] = None

        if cfg.auto_classify:
            try:
                classification = self.classify_image(
                    uploaded_name,
                    ocr_id=0,
                    lang_type=0,
                    reading_direction=0,
                )
            except RateLimitError:
                raise
            except APIError as exc:
                # Fall back to configured OCR settings instead of discarding the uploaded image.
                classification = {"error": str(exc)}
            else:
                detected_ocr_id = classification.get("ocr_id")
                detected_lang_type = classification.get("lang_type")
                detected_direction = classification.get("reading_direction")

                if detected_ocr_id is not None:
                    try:
                        ocr_id = int(detected_ocr_id)
                    except (TypeError, ValueError):
                        pass
                if detected_lang_type is not None:
                    try:
                        lang_type = int(detected_lang_type)
                    except (TypeError, ValueError):
                        pass
                if detected_direction is not None:
                    try:
                        reading_direction = int(detected_direction)
                    except (TypeError, ValueError):
                        pass

                # Reuse inline OCR results if structure-classification already returned them
                # (mirrors kimhannom.fit.hcmus.edu.vn/js/hannom.js).
                inline_result = classification.get("result")
                if (
                    isinstance(inline_result, dict)
                    and inline_result.get("result_bbox")
                    and inline_result.get("result_ocr_text")
                    and ocr_id in {1, 2, 4, 5}
                    and lang_type != 1
                ):
                    data = inline_result

                # Normalize structure-classification ocr_id values for /image-ocr.
                if ocr_id == 4:
                    ocr_id = 1
                elif ocr_id == 5:
                    epitaph = 1
                    ocr_id = 1
                elif ocr_id == 6:
                    ocr_id = 1

        if data is None:
            data = self.ocr_image(
                uploaded_name,
                ocr_id=ocr_id,
                lang_type=lang_type,
                reading_direction=reading_direction,
                epitaph=epitaph,
                tag=cfg.tag,
            )

        lines = self._extract_ocr_lines(data.get("result_ocr_text"))
        return {
            "text": "\n".join(lines),
            "lines": lines,
            "file_name": uploaded_name,
            "result_file_name": data.get("result_file_name"),
            "result_bbox": data.get("result_bbox") or [],
            "classification": classification,
        }

    @staticmethod
    def _extract_ocr_lines(raw_lines: Any) -> list[str]:
        """Normalize result_ocr_text from /image-ocr or /structure-classification."""
        if not raw_lines:
            return []
        if isinstance(raw_lines, str):
            raw_lines = [raw_lines]
        lines: list[str] = []
        for item in raw_lines:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
                text = str(item[1][0]).strip()
            else:
                text = str(item).strip()
            if text:
                lines.append(text)
        return lines

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HanNomAPIClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
