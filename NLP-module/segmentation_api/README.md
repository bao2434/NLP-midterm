# `segmentation_api` — Kim Han-Nom Sentence Segmentation API Module

Discovers all text files in a folder, divides each non-empty text file into chunks where each chunk is a non-empty line before a newline (respecting the API's 10,000-token maximum input length), sends each chunk sequentially to the [Kim Han-Nom API](https://kimhannom.clc.hcmus.edu.vn/) (`/api/web/clc-sinonom/separate-sentences` or `/api/web/clc-sinonom/sentence-punctuation`), and concatenates the segmented results (adding a newline whenever a chunk result does not end with one so two sentences are never joined on the same line) into the output directory (`<output-dir>/<doc_name>.txt`).

## Traditional Medicine Defaults

- **Default `genre`**: `medical` (Y học) — matches the `medical` genre supported by `/separate-sentences` and `/sentence-punctuation`.
- **Default `endpoint`**: `separate-sentences` with `punctuation_group="all"`.
- Every discovered `.txt` file in `--input-dir` is split into non-empty line chunks, sent sequentially to the API, and concatenated with guaranteed newline boundaries before writing to `--output-dir`.

## Dependencies

Install dependencies from `segmentation_api/requirements.txt`:

```bash
pip install -r segmentation_api/requirements.txt
```

## API Rules & Rate-Limit Modes (`limited` vs `--unlimited`)

Per `TaiLieuAPI_HanNom_CLC.docx`:

- **Base domain**: `https://kimhannom.clc.hcmus.edu.vn`
- **Rate limits**: `25` requests/minute, `50` requests/hour, `200` requests/day, `600` requests/week.
- **User-Agent rule**: `User-Agent` must **not** start with `python-requests`.
- **Authentication**: Log in to `https://kimhannom.fit.hcmus.edu.vn/`, open **Inspect -> Storage -> Cookies**, copy the `token` cookie, and pass it via `--token` or `HANNOM_API_TOKEN`.

### API Call Limit Modes

- **Restricted (`--rate-limit-mode limited`, default)**: Enforces local sliding-window caps (`--max-requests-per-minute`, `--max-requests-per-hour`, `--max-requests-per-day`, `--max-requests-per-week`), optional `--max-requests` per run, and a default `--delay 3.0` seconds between requests.
- **Unlimited (`--unlimited` or `--rate-limit-mode unlimited`)**: Calls the API continuously and sequentially without caring about limits — disables all local request-count caps, quota-preserving early stops, and sets the default inter-request delay to `0.0`s.

Already-written non-empty files in `--output-dir` are skipped on re-run unless `--no-resume` is specified.

## Usage (`main.py`)

```bash
# 1. Dry-run to inspect all discovered text files
python main.py --input-dir ./refined_text --dry-run

# 2. Run continuously and sequentially without caring about limits
python main.py \
  --input-dir ./refined_text \
  --output-dir ./seg_api_output \
  --token "<YOUR_COOKIE_TOKEN>" \
  --genre medical \
  --unlimited

# 3. Run in limited mode with rate-limit guards and delay
python main.py \
  --input-dir ./refined_text \
  --output-dir ./seg_api_output \
  --token "<YOUR_COOKIE_TOKEN>" \
  --genre medical \
  --rate-limit-mode limited \
  --delay 3.0

# 4. Only keep sentence-ending punctuation (。？！)
python main.py \
  --input-dir ./refined_text \
  --output-dir ./seg_api_output \
  --token "<YOUR_COOKIE_TOKEN>" \
  --genre medical \
  --punctuation-group fullstop \
  --unlimited
```
