"""
Quick test for the Kim Han Nom sentence-alignment API (Han <-> Vietnamese).

Usage:
    pip install requests
    python test_sentence_alignment.py --token <token>

    # or via environment variable
    #   PowerShell:  $env:KHN_TOKEN = "<token>"
    #   cmd:         set KHN_TOKEN=<token>
    #   bash:        export KHN_TOKEN="<token>"
"""
import argparse
import json
import os
import sys
import time

import requests

BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
ENDPOINT = "/api/web/clc-sinonom/sentence-alignment"

# NOTE: User-Agent must NOT start with "python-requests"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
}

HAN = "洪德十八年冬十二月。\n帝遣阮仁政徃福州。"
VI = ("Tháng mười hai mùa đông năm Hồng Đức thứ 18.\n"
      "Nhà vua sai Nguyễn Nhân Chính đến Phúc Châu.")

REQUIRED_KEYS = {"han_start", "han_end", "vi_start", "vi_end",
                 "han_text", "vietnamese_text", "score"}


def call(han, vi):
    r = requests.post(BASE_URL + ENDPOINT, headers=HEADERS,
                      json={"han": han, "vi": vi}, timeout=60)
    try:
        body = r.json()
    except ValueError:
        body = None
    return r, body


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    return cond


def test_basic():
    print("\n=== basic alignment (2 Han lines <-> 2 Vietnamese lines) ===")
    start = time.time()
    r, body = call(HAN, VI)
    print(f"  HTTP {r.status_code}, {time.time() - start:.2f}s")
    if body is None:
        return check("response is JSON", False, r.text[:200])

    ok = check("HTTP 200 + success", r.status_code == 200
               and body.get("is_success") and body.get("code") == "000000",
               f"(code={body.get('code')})")
    if not ok:
        print("  message:", json.dumps(body.get("message"), ensure_ascii=False))
        return False

    result = body["data"]["result"]
    print(json.dumps(result, ensure_ascii=False, indent=2))

    ok &= check("2 aligned pairs returned", len(result) == 2, f"(got {len(result)})")
    ok &= check("each item has expected keys",
                all(REQUIRED_KEYS <= set(item) for item in result))
    ok &= check("pairs are 1-to-1 in order",
                [(i["han_start"], i["vi_start"]) for i in result] == [(0, 0), (1, 1)])
    ok &= check("han_text matches input lines",
                [i["han_text"] for i in result] == HAN.split("\n"))
    ok &= check("scores are floats",
                all(isinstance(i["score"], (int, float)) for i in result))
    return ok


def test_empty_input():
    print("\n=== empty input (expect a validation error) ===")
    r, body = call("", "")
    print(f"  HTTP {r.status_code}, body: {json.dumps(body, ensure_ascii=False)[:300] if body else r.text[:200]}")
    # Server returns HTTP 200 with is_success=false, code 000100 (field required)
    return (check("no server error (HTTP < 500)", r.status_code < 500)
            and check("rejected with is_success=false",
                      body is not None and body.get("is_success") is False
                      and body.get("code") != "000000",
                      f"(code={body.get('code') if body else None})"))


def main():
    parser = argparse.ArgumentParser(description="Test the sentence-alignment API")
    parser.add_argument("--token", default=os.environ.get("KHN_TOKEN", ""),
                        help="Bearer token (or set env var KHN_TOKEN)")
    args = parser.parse_args()

    token = args.token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        HEADERS["Authorization"] = f"Bearer {token}"
        print("Using token:", token[:6] + "..." + token[-4:])
    else:
        print("WARNING: no token given (--token or KHN_TOKEN).")

    results = {"basic": test_basic()}
    time.sleep(3)  # rate limit: 25/min
    results["empty input"] = test_empty_input()

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
