"""
Client for the Kim Han Nom Chinese–Vietnamese sentence-alignment API.

API documented in TaiLieuAPI_HanNom_CLC.txt:
    POST /api/web/clc-sinonom/sentence-alignment

Install:
    pip install requests

Quick start:
    from hanviet_alignment import HanVietAlignmentClient

    with HanVietAlignmentClient() as client:
        result = client.align(
            han="洪德十八年冬十二月。\n帝遣阮仁政徃福州。",
            vi=(
                "Tháng mười hai mùa đông năm Hồng Đức thứ 18.\n"
                "Nhà vua sai Nguyễn Nhân Chính đến Phúc Châu."
            ),
        )

        for item in result:
            print(item.han_text, " <-> ", item.vietnamese_text)
            print("score:", item.score)

Notes:
- The API documentation specifies a 25-requests/minute limit and asks clients
  to delay requests and avoid concurrent bursts. The default 2.5-second
  minimum interval respects that per-minute limit. The documented hourly,
  daily, and weekly quotas are lower for sustained use, so do not treat this
  client-side delay as permission to exceed those quotas.
- Authorization is optional here because the documented cURL example for this
  endpoint does not include it. If the service requires login for your account,
  pass your website token via token=... or HAN_NOM_API_TOKEN.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import requests


DEFAULT_BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
ALIGNMENT_PATH = "/api/web/clc-sinonom/sentence-alignment"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)


class HanNomAPIError(RuntimeError):
    """Base exception for errors returned by or communicating with the API."""


class HanNomHTTPError(HanNomAPIError):
    """The server returned a non-successful HTTP status."""

    def __init__(self, status_code: int, message: str, response_text: str = ""):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"HTTP {status_code}: {message}")


class HanNomResponseError(HanNomAPIError):
    """The server returned an invalid or unsuccessful API response."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        api_message: Any = None,
        payload: Any = None,
    ):
        self.code = code
        self.api_message = api_message
        self.payload = payload
        detail = f"API error {code}: {message}" if code else message
        super().__init__(detail)


@dataclass(frozen=True)
class Alignment:
    """One aligned span returned by the sentence-alignment endpoint.

    The start/end indices are preserved exactly as returned by the API.
    The documentation's example uses zero-based sentence indices.
    """

    han_start: int
    han_end: int
    vi_start: int
    vi_end: int
    han_text: str
    vietnamese_text: str
    score: float

    @classmethod
    def from_api(cls, item: Mapping[str, Any]) -> "Alignment":
        required = (
            "han_start",
            "han_end",
            "vi_start",
            "vi_end",
            "han_text",
            "vietnamese_text",
            "score",
        )
        missing = [key for key in required if key not in item]
        if missing:
            raise HanNomResponseError(
                f"Alignment item is missing required fields: {missing}",
                payload=dict(item),
            )

        try:
            return cls(
                han_start=int(item["han_start"]),
                han_end=int(item["han_end"]),
                vi_start=int(item["vi_start"]),
                vi_end=int(item["vi_end"]),
                han_text=str(item["han_text"]),
                vietnamese_text=str(item["vietnamese_text"]),
                score=float(item["score"]),
            )
        except (TypeError, ValueError) as exc:
            raise HanNomResponseError(
                f"Invalid alignment item: {exc}",
                payload=dict(item),
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    """Parsed response for one alignment request."""

    alignments: tuple[Alignment, ...]
    raw_response: Mapping[str, Any]

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.alignments]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignments": self.to_list(),
            "raw_response": dict(self.raw_response),
        }

    def as_pairs(self) -> list[tuple[str, str]]:
        """Return [(Chinese span, Vietnamese span), ...]."""
        return [
            (item.han_text, item.vietnamese_text)
            for item in self.alignments
        ]

    def to_text(self, separator: str = "\n") -> str:
        """Format the alignment as readable paired lines."""
        return separator.join(
            f"{item.han_text}\t{item.vietnamese_text}"
            for item in self.alignments
        )


class HanVietAlignmentClient:
    """Reusable client for the Hán–Việt sentence-alignment API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 60.0,
        min_interval: float = 2.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if min_interval < 0:
            raise ValueError("min_interval cannot be negative")
        if not user_agent.strip():
            raise ValueError("user_agent cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + ALIGNMENT_PATH
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request_time: Optional[float] = None

        self.session = session or requests.Session()
        self._owns_session = session is None

        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        token = token or os.getenv("HAN_NOM_API_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "HanVietAlignmentClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _wait_for_rate_limit(self) -> None:
        if self.min_interval == 0 or self._last_request_time is None:
            return

        elapsed = time.monotonic() - self._last_request_time
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _parse_response(response: requests.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise HanNomResponseError(
                "The API did not return valid JSON.",
                payload=response.text,
            ) from exc

        if not isinstance(payload, dict):
            raise HanNomResponseError(
                "Expected the API response to be a JSON object.",
                payload=payload,
            )

        if payload.get("is_success") is not True or payload.get("code") != "000000":
            message = payload.get("message")
            if isinstance(message, dict):
                message_text = (
                    message.get("text")
                    or message.get("title")
                    or str(message)
                )
            else:
                message_text = str(message or "The API reported an unsuccessful request.")

            raise HanNomResponseError(
                message_text,
                code=str(payload.get("code")) if payload.get("code") is not None else None,
                api_message=message,
                payload=payload,
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise HanNomResponseError(
                "Successful response does not contain a data object.",
                payload=payload,
            )

        return payload

    def align(self, han: str, vi: str) -> AlignmentResult:
        """Align a Chinese text with its Vietnamese translation.

        Args:
            han: Chinese/Hán source text.
            vi: Vietnamese translation of the source text.

        Returns:
            AlignmentResult containing the aligned spans and scores.

        Raises:
            ValueError: If either input is empty or not a string.
            HanNomHTTPError: If the server returns a non-2xx HTTP status.
            HanNomResponseError: If the API response is unsuccessful or malformed.
            requests.RequestException: For network-level errors/timeouts.
        """
        if not isinstance(han, str) or not han.strip():
            raise ValueError("'han' must be a non-empty string")
        if not isinstance(vi, str) or not vi.strip():
            raise ValueError("'vi' must be a non-empty string")

        self._wait_for_rate_limit()
        self._last_request_time = time.monotonic()

        try:
            response = self.session.post(
                self.endpoint,
                json={"han": han, "vi": vi},
                timeout=self.timeout,
            )
        except requests.RequestException:
            # Preserve requests' original exception type for callers.
            raise

        if not response.ok:
            raise HanNomHTTPError(
                response.status_code,
                response.reason or "Request failed",
                response.text,
            )

        payload = self._parse_response(response)
        data = payload["data"]
        raw_alignments = data.get("result")

        if not isinstance(raw_alignments, list):
            raise HanNomResponseError(
                "Response data.result is missing or is not a list.",
                payload=payload,
            )

        alignments = tuple(
            Alignment.from_api(item)
            for item in raw_alignments
            if isinstance(item, dict)
        )
        if len(alignments) != len(raw_alignments):
            raise HanNomResponseError(
                "One or more alignment entries are not JSON objects.",
                payload=payload,
            )

        return AlignmentResult(
            alignments=alignments,
            raw_response=payload,
        )

    def align_many(
        self,
        items: Iterable[tuple[str, str]],
        *,
        continue_on_error: bool = False,
    ) -> Iterator[AlignmentResult | Exception]:
        """Align multiple (han, vi) pairs sequentially.

        Sequential execution is intentional: it avoids sending a burst of
        concurrent requests to the service.

        If continue_on_error=False, the first exception is raised.
        If True, exceptions are yielded in place and processing continues.
        """
        for han, vi in items:
            try:
                yield self.align(han, vi)
            except Exception as exc:
                if not continue_on_error:
                    raise
                yield exc


def align_text(
    han: str,
    vi: str,
    *,
    token: Optional[str] = None,
    min_interval: float = 2.5,
) -> AlignmentResult:
    """Convenience function for a single request."""
    with HanVietAlignmentClient(
        token=token,
        min_interval=min_interval,
    ) as client:
        return client.align(han, vi)


__all__ = [
    "Alignment",
    "AlignmentResult",
    "HanNomAPIError",
    "HanNomHTTPError",
    "HanNomResponseError",
    "HanVietAlignmentClient",
    "align_text",
]
