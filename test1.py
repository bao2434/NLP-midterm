"""
Translate a Chinese passage to Vietnamese with the Kim Han Nom API.

Usage:
    pip install requests
    python test_translate_passage.py --token <token>
    python test_translate_passage.py --token <token> --split          # one request per line
    python test_translate_passage.py --token <token> --endpoint admin # admin / verse / prose

    # Optional: the API is built for classical Han (traditional characters).
    # Your passage is modern SIMPLIFIED Chinese, so converting first may help:
    pip install opencc-python-reimplemented
    python test_translate_passage.py --token <token> --traditional
"""
import argparse
import json
import os
import sys
import time

import requests

BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
DELAY = 3  # seconds between requests (limit: 25/min, 50/hour, 200/day)

ENDPOINTS = {
    "prose": ("/api/web/clc-sinonom/sinonom-fair-seg", False),
    "admin": ("/api/web/clc-sinonom/sinonom-administration-translation", True),
    "verse": ("/api/web/clc-sinonom/sinonom-verse-translation", True),
}  # name -> (path, needs font_type)

HEADERS = {
    # must NOT start with "python-requests"
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
}

PASSAGE = """体验——中医入门之道体验，以体处之，以心验之，是中国古人认识世界的重要方法。
老子平日里观察自然，思考人性，探寻道。
一天，老子路经溪流，看着潺潺的溪水，陷入沉思。
过了一会儿，若有所悟，感慨道：“上善若水。”
老子是怎样得出这个结论的呢？
“水善利万物而不争，处众人之所恶，故几于道。”
水从来都是从高处往低处流，人却都是要从低处往高处走的；
水到的地方，往往是人所不喜、不屑、不齿的地方。
水来到这里，并不是为消磨时光，而是为了濡养、滋助万物的生长。
水在“利万物”之后，却不见它对万物有任何索取，要求任何回报。
这样的一种性情，已经与“道”很接近了，是为上善。
上面是一个典型的“体验”的例子。
这里，老子之体在溪水边，检验的是水之性。
以身体处水旁，我与水融为一体，以我心验水性，即是体验。"""


def translate(text, endpoint):
    path, needs_font = ENDPOINTS[endpoint]
    payload = {"text": text, "lang_type": 1}
    if needs_font:
        payload["font_type"] = 1
    r = requests.post(BASE_URL + path, headers=HEADERS, json=payload, timeout=120)
    try:
        body = r.json()
    except ValueError:
        return False, f"HTTP {r.status_code}, non-JSON: {r.text[:200]}"
    if r.status_code == 200 and body.get("is_success") and body.get("code") == "000000":
        return True, body["data"]["result"]
    return False, f"HTTP {r.status_code}, code={body.get('code')}, " \
                  f"message={json.dumps(body.get('message'), ensure_ascii=False)}"


def to_traditional(text):
    try:
        from opencc import OpenCC
    except ImportError:
        sys.exit("Install converter first: pip install opencc-python-reimplemented")
    return OpenCC("s2t").convert(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("KHN_TOKEN", ""))
    ap.add_argument("--endpoint", choices=ENDPOINTS, default="prose")
    ap.add_argument("--split", action="store_true", help="send one request per line")
    ap.add_argument("--traditional", action="store_true",
                    help="convert simplified -> traditional before sending")
    args = ap.parse_args()

    token = args.token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        HEADERS["Authorization"] = f"Bearer {token}"
        print("Using token:", token[:6] + "..." + token[-4:])
    else:
        print("WARNING: no token given (--token or KHN_TOKEN).")

    text = to_traditional(PASSAGE) if args.traditional else PASSAGE
    print(f"Endpoint: {args.endpoint} | split={args.split} | traditional={args.traditional}")

    chunks = [l for l in text.split("\n") if l.strip()] if args.split else [text]
    failures = 0
    for i, chunk in enumerate(chunks):
        if i:
            time.sleep(DELAY)
        start = time.time()
        ok, res = translate(chunk, args.endpoint)
        print(f"\n[{i + 1}/{len(chunks)}] {'PASS' if ok else 'FAIL'} ({time.time() - start:.2f}s)")
        print("  ZH:", chunk if len(chunk) < 120 else chunk[:120] + "...")
        if ok:
            print("  VI:", "\n      ".join(res))
        else:
            failures += 1
            print("  ERR:", res)

    print(f"\n=== {len(chunks) - failures}/{len(chunks)} requests succeeded ===")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
