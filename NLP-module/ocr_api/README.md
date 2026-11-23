# `ocr_api` — Kim Han-Nom API OCR Module

Detects PDFs in a folder, renders each page, calls the [Kim Han-Nom API](https://kimhannom.fit.hcmus.edu.vn/), saves incremental JSONL checkpoints for resumability, and exports per-book `.txt` files.

## Required Libraries & Kaggle Environment Notes

| Library | Purpose | Pre-installed on Kaggle? |
| --- | --- | --- |
| `pypdfium2` | Render PDF pages to PIL images (alternative fallback: `pymupdf` / `fitz`) | **No — must be installed** |
| `requests` | HTTP calls to the Kim Han-Nom API | Usually pre-installed |
| `pillow` (`PIL`) | Image resizing and JPEG/PNG encoding | Usually pre-installed |

### Installing on Kaggle

Kaggle notebook environments typically include `requests` and `Pillow`, but **do not include `pypdfium2`** (or `pymupdf`). Run the following cell before executing `main.py`:

```bash
!pip install -q pypdfium2
```

Or install all dependencies from `ocr_api/requirements.txt`:

```bash
!pip install -q -r ocr_api/requirements.txt
```

## API Rules & Why Processing Is Sequential (No Parallelism)

Per `TaiLieuAPI_HanNom_CLC .docx`:

- **Rate limits**: `25` requests/minute, `50` requests/hour, `200` requests/day, `600` requests/week.
- **Congestion rule**: Callers must add a delay between requests and avoid sending concurrent bursts.
- **User-Agent rule**: `User-Agent` must **not** start with `python-requests`.
- **Authentication**: Log in to `https://kimhannom.fit.hcmus.edu.vn/`, open **Inspect -> Storage -> Cookies**, copy the `token` cookie, and pass it via `--token` or `HANNOM_API_TOKEN`.

Because of these limits, `ocr_api` intentionally runs **sequentially** with a default `--delay 3.0` seconds between requests, a sliding-window rate-limit guard (`--rate-limit-mode limited`), and an early-stop guard (`--max-consecutive-errors 5`) so repeated API errors don't burn through your hourly quota. Every completed page is immediately appended to `{output_dir}/{book}/api_{mode}.jsonl` so runs stopped by rate limits or time budgets resume cleanly on the next run.

### API Call Limit Modes (`limited` vs `unlimited`)

`ocr_api` supports two call-limit modes via `APIConfig(rate_limit_mode=...)` or CLI flags:

- **Restricted (`--rate-limit-mode limited`, default)**: Enforces local sliding-window caps (`--max-requests-per-minute`, `--max-requests-per-hour`, `--max-requests-per-day`, `--max-requests-per-week`) and optional `--max-requests` per run. Setting any individual cap to `0` disables that specific window.
- **Unlimited (`--rate-limit-mode unlimited` or `--unlimited`)**: Disables all local request-count limits (per-minute/hour/day/week and `--max-requests`) to allow unlimited API calls (e.g., for privileged/internal tokens). Combine with `--delay 0` if you also want to remove the inter-request pause.

> **Tip:** Avoid passing `--auto-classify` unless your PDFs mix document types/directions. `--auto-classify` calls `/api/web/clc-sinonom/structure-classification` (+1 request/page), reducing throughput from 25 pages/hour to ~16 pages/hour under the 50 req/hour limit.

## Usage (`main.py`)

```bash
# 1. Dry-run to inspect discovered PDFs and estimated API requests
python main.py --input-dir . --pages 1-5 --dry-run

# 2. Run Han-Nom OCR with restricted limits (default: 25/min, 50/hr, 200/day, 600/wk)
python main.py \
  --input-dir /kaggle/input \
  --output-dir /kaggle/working/ocr_api_raw \
  --txt-dir /kaggle/working/ocr_api_txt \
  --token "<YOUR_COOKIE_TOKEN>" \
  --mode hannom \
  --rate-limit-mode limited \
  --max-requests-per-minute 25 \
  --max-requests-per-hour 50 \
  --max-requests-per-day 200 \
  --max-requests-per-week 600 \
  --ocr-id 1 \
  --lang-type 1 \
  --reading-direction 1 \
  --delay 3.0

# 3. Run with unlimited API calls (disables local rate-limit & per-run caps)
python main.py \
  --input-dir /kaggle/input \
  --output-dir /kaggle/working/ocr_api_raw \
  --txt-dir /kaggle/working/ocr_api_txt \
  --token "<YOUR_COOKIE_TOKEN>" \
  --mode hannom \
  --unlimited \
  --delay 0

# 4. Run Vietnamese OCR (/viet-ocr)
python main.py \
  --input-dir /kaggle/input \
  --output-dir /kaggle/working/ocr_api_raw \
  --txt-dir /kaggle/working/ocr_api_txt \
  --token "<YOUR_COOKIE_TOKEN>" \
  --mode viet \
  --delay 3.0
```
