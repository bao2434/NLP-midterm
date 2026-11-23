# Text Refinement Documentation (`./text` -> `./refined_text`)

This document records all preprocessing rules, automated transformations, quantitative metrics, and concrete before/after examples applied to the 10 text files in `./text` to produce the cleaned corpus in `./refined_text`.

- **Pipeline Script**: [`clean_texts.py`](./clean_texts.py)
- **Quantitative Stats (JSON)**: [`refinement_stats.json`](./refinement_stats.json)
- **Input Folder**: `./text` (10 `.txt` files, UTF-8)
- **Output Folder**: `./refined_text` (10 `.txt` files, UTF-8, `\n` line endings)

---

## 1. Summary of Quantitative Changes

| File | Dominant Script | Input Lines | Output Lines | Input Chars | Output Chars | Front/Back Matter & Non-Main Removed | Headers / Footers / Page Nums Removed | Footnotes Removed (Lines + Inline) | Tables / Plates Removed | Dup Lines Removed | Page-Break Merges | Cross-Script Removed | Garbage Tokens Cleaned | Punct/WS Lines Normalized |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nam-duoc-than-hieu-pdf.txt` | Vietnamese | 12,141 | 11,338 | 1,051,267 | 1,002,115 | 498 lines | 78 | 125 lines + 199 inline | 25 lines | 5 | 36 | 0 *(18 CJK on removed cover)* | 4,495 | 303 |
| `nam_khoa_nu_khoa.txt` | Vietnamese | 3,356 | 2,671 | 258,055 | 216,036 | 488 lines | 83 | 399 inline (+347 endnote lines in back matter) | 0 | 1 | 0 | 1,398 Han-Nom spans | 95 | 333 |
| `Pho_Thanh_Chu_Nu_Khoa.txt` | Classical Chinese | 3,675 | 3,640 | 54,204 | 53,295 | 0 | 0 | 21 lines + 10 inline | 0 | 2 | 0 | 0 | 72 | 61 |
| `Linh khởi điểm học trung y..._resolved.txt` | Simplified Chinese | 393 | 87 | 139,465 | 131,831 | 17 pages | 2 | 0 | 0 | 0 | 90 | 0 | 74 | 7 |
| `phuong-te-hoc..._resolved.txt` | Simplified Chinese | 531 | 111 | 301,500 | 265,202 | 38 pages | 0 | 99 inline | 5 blocks (+10 appendix table pages) | 0 | 116 | 0 | 15 | 35 |
| `Tân_san_Nam_dược..._q5_resolved.txt` | Han-Nom / Classical Chinese | 101 | 10 | 7,855 | 7,769 | 0 | 1 | 0 | 0 | 0 | 41 | 0 | 4 | 0 |
| `Tân_san_Nam_dược..._q6_resolved.txt` | Han-Nom / Classical Chinese | 299 | 41 | 22,950 | 22,712 | 0 | 0 | 0 | 0 | 0 | 109 | 0 | 21 | 0 |
| `Tân_san_Nam_dược..._q8_resolved.txt` | Han-Nom / Classical Chinese | 223 | 21 | 19,852 | 19,654 | 1 page | 2 | 0 | 0 | 0 | 90 | 0 | 11 | 0 |
| `Tân_san_Nam_dược..._q9_resolved.txt` | Han-Nom / Classical Chinese | 259 | 23 | 21,309 | 20,978 | 2 pages | 1 | 0 | 0 | 0 | 105 | 0 | 13 | 0 |
| `Tân_san_Nam_dược..._q10_resolved.txt` | Han-Nom / Classical Chinese | 261 | 26 | 22,847 | 22,586 | 1 page | 19 | 0 | 0 | 0 | 104 | 0 | 11 | 0 |

---

## 2. Requirements Coverage & Detailed Rules Applied

### 2.1 Remove headers, footers, and non-main-content pages
- **`nam-duoc-than-hieu-pdf.txt`**:
  - Removed front matter (L1–L75): cover pages, publisher imprint, watermark URLs (`https://vnras.com/drug/`), temple photograph caption (`Miếu Nghe, thơ Tuệ Tình ở chùa Giam xã Cẩm Sơn`), `LỜI NÓI ĐẦU`, `TIỂU SỬ CỦA TUỆ TĨNH`, `LỜI TỰA`, and `PHÀM LỆ`. Main content begins cleanly at `NAM DƯỢC THẦN HIỆU` / `QUYỂN ĐẦU`.
  - Removed internal table of contents `THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ MỤC LỤC` (41 lines).
  - Removed back matter (L11760–L12141): `MỤC LỤC`, print colophon, and back cover price/address.
  - Removed 78 standalone page numbers (e.g., `15`, `31`, `47`, `61`, ...) and watermark URLs (`https://vnras.com/drug/`).
- **`nam_khoa_nu_khoa.txt`**:
  - Removed translator credit header line 1 (`DỊCH GIÁ: ĐỊNH NINH - LỄ ĐỨC THIẾP`).
  - Removed trailing `HÉT`, the entire `PHỤ CHÚ về NỮ KHOA của DỊCH GIẢ` endnotes section, `MỤC LỤC NỮ KHOA`, and publisher/permit/price colophon (487 lines).
  - Removed 83 decorative section dividers (`oOo`, `O Oo`, `o0o`).
- **`Linh khởi điểm học trung y_ đái nhĩ thể nghiệm trung y đích 12 đường khóa_resolved.txt`**:
  - Removed 12 front-matter pages: cover, back-cover blurb, title page, CIP/ISBN copyright page, prefaces (`序`, `前言`), and `目录` (TOC). Main content starts at `体验——中医入门之道`.
  - Removed 5 back-matter pages: `附录C网友学习感言` (reader testimonials) and `后记` (postscript).
  - Removed forum watermarks (`更多精彩欢迎访问：中医中药论坛：http：//bbs.zhong-yao.net`) and running header (`——带你体验中医的12堂课`).
- **`phuong-te-hoc-cung-trung-y-trung-duoc-cham-cuu-chuyen-nghiep-dung-hua_resolved.txt`**:
  - Removed 8 front-matter pages: cover, `前言`, `编写说明`, and `目录` (TOC). Kept `绪言` and `上篇总论` through end of main text (`小结 痈疡剂`).
  - Removed 30 back-matter pages: `附录1.常用中成药简表` (multi-page tables), `附录2` (mnemonic songs with running headers `附录`/`附錄`/`附致`), and stroke-order index (`四画`...`二十三画`).
- **`Tân_san_Nam_dược_thần_hiệu_thập_khoa_ứng_trị_q5, q6, q8, q9, q10`**:
  - Removed cover-only title pages (`卷捌`, `卷玖`, `卷拾`) and the q9 disease-list TOC page (`卷九小兒科息驚慢驚...`).
  - Removed 23 woodblock centerfold folio page numbers printed at the start of pages (e.g., `九`, `十一`, `十二`, `六四`, `六五`, `四13`) while preserving all recipe openers (`一經治`, `一方`, `三開`, `五疸`).

### 2.2 Merge words/sentences broken across page breaks
- **Page-per-line Chinese OCR files** (`Linh khởi điểm...`, `phuong-te-hoc...`, `Tân_san_Nam_dược... q5–q10`):
  - Rejoined sentences cut mid-word or mid-clause across page boundaries (655 merges total across these 7 files).
  - For modern Chinese texts, pages not ending in terminal punctuation (`。！？；：”’）】》.!?;:`) are merged seamlessly into the next page without inserting stray spaces.
  - For unpunctuated classical woodblock texts (`Tân_san_Nam_dược...`), pages are merged continuously except where a new recipe opener (`一經治`, `一傳治`, `又方`, `又經方`) follows a natural recipe ending (`效`, `愈`, `止`, `消`, `散`, `妙`, `安`, `瘥`, `良`, `。`).
- **Vietnamese OCR files** (`nam-duoc-than-hieu-pdf.txt`, `nam_khoa_nu_khoa.txt`):
  - Merged prose lines broken mid-sentence across page/line wraps (36 merges in `nam-duoc-than-hieu-pdf.txt`) when a prose line lacks terminal punctuation and the next line begins with a lowercase letter, while preserving headings, poetry lines, and herb ingredient lists.

### 2.3 Remove duplicate lines from OCR overlapping regions
- Applied exact and high-similarity (`>= 0.92` SequenceMatcher ratio for lines `>= 25` chars) adjacent-line deduplication within a sliding window.
- Removed 8 duplicate OCR lines (`nam-duoc-than-hieu-pdf.txt`: 5, `Pho_Thanh_Chu_Nu_Khoa.txt`: 2, `nam_khoa_nu_khoa.txt`: 1).

### 2.4 Remove footnotes outside main content and inline footnote markers
- **`nam-duoc-than-hieu-pdf.txt`**:
  - Removed 125 bottom-of-page footnote lines (both `(1) ...` style and editorial `1. Nguyên bản ...` / botanical synonym footnotes, carefully distinguishing them from numbered herb/prescription entries).
  - Removed 199 inline footnote indices (`(1)`–`(99)`, `[1]`, `$ ^{3} $`, `$ ^{(1)} $`, `$ ^{6} $`, etc.).
- **`nam_khoa_nu_khoa.txt`**:
  - Removed the entire translator endnotes section (`PHỤ CHÚ về NỮ KHOA của DỊCH GIẢ`, L2872–L3218).
  - Removed 399 inline footnote markers (`(1)`–`(80)`, `$ ^{2} $`, and `(Xem phụ chú Nam Khoa)`).
- **`Pho_Thanh_Chu_Nu_Khoa.txt`**:
  - Removed 21 marginal commentary (`眉批︰`) lines outside the main text and 10 trailing editorial section indices (`（29）`–`（38）`).
- **`phuong-te-hoc...`**:
  - Removed 99 inline LaTeX superscript citation markers (`$^{1}$`, `$^{2}$`, `$^{3}$`) after quotations/sentences.
  - Converted structural list superscripts in `臣药/佐药/使药` definitions (`有两种意义。$^{1}$...$^{2}$...`) into `①`, `②`, `③` so no meaning is lost.

### 2.5 Remove cross-script text (Han-Nom in Vietnamese-dominant files and vice versa)
- **Vietnamese-dominant files** (`nam-duoc-than-hieu-pdf.txt`, `nam_khoa_nu_khoa.txt`):
  - Stripped all Han-Nom / CJK ideographs and CJK-specific symbols (`\u2e80-\u2fdf`, `\u3005-\u3007`, `\u3400-\u4dbf`, `\u4e00-\u9fff`, `\uf900-\ufaff`, `\U00020000-\U0003134f`), along with any empty parentheses/brackets or punctuation-only lines left behind.
  - In `nam_khoa_nu_khoa.txt`, removed **1,398 Han-Nom spans** (including standalone Han section headers such as `調經`, `經水先期` and inline Han herb/formula names such as `THANH KINH THANG 清經湯` -> `THANH KINH THANG`, `青萬 THANH CAO 2 đồng cản` -> `THANH CAO 2 đồng cản`).
  - In `nam-duoc-than-hieu-pdf.txt`, all 18 Han characters on the front cover were removed with front matter, and the pass guarantees 0 remaining Han-Nom characters in the body.
- **Chinese / Han-Nom-dominant files** (all 8 remaining files):
  - Applied `strip_vietnamese_in_cjk` to remove any Vietnamese words (tokens containing Vietnamese diacritics or `đ`/`Đ`) while strictly preserving measurement/scientific units (`g`, `kg`, `ml`, `°C`, `II`, etc.). All 8 files verified to contain **0** Vietnamese diacritic characters.

### 2.6 Fix garbage characters and abnormal whitespace
- Normalized all files to Unicode **NFC** (fixing decomposed combining diacritics such as `́`, `̀`, `̣`, `̉`, `̃` in Vietnamese OCR).
- Cleaned LaTeX/Markdown OCR artifacts:
  - `$\underset{\cdot}{及}$` -> `及`, `$\underset{\cdot}{影}\underset{\cdot}{响}$` -> `影响`
  - `$37^{\circ}C$` -> `37°C`, `$40°C` -> `40°C`, `\sim` -> `~`
  - `$ \textit{hòe} $` -> `hòe`, `$ \textbf{...} $` -> `...`, `$ \text{...} $` -> `...`
  - Stripped stray `*`, `\`, `{`, `}`, `^`, `$`, `\n` literals, OCR table pipes (`|富歸|ĐƯƠNG QUY|2 dòng` -> `ĐƯƠNG QUY 2 dòng`), decorative `◎`, and `思考§` / `恩考§` -> `思考：`.
- Removed 60 stray Bopomofo/Kana OCR garbage characters (`ㄐ`, `ㄔ`, `ㄇ`, `ㄟ`, `ㄢ`, `ㄘ`, `ㄗ`, `ㄡ`, `ㄏ`, `り`) in `Tân_san_Nam_dược...` woodblock scans.
- Collapsed multiple spaces/tabs, trimmed trailing spaces, removed erroneous spaces between CJK characters, and collapsed excessive blank lines.

### 2.7 Normalize punctuation without changing meaning
- **Vietnamese**: converted `« »`, `《 》`, `“ ”` -> `" "`; normalized fullwidth CJK punctuation (`，`, `：`, `；`, `（`, `）`, `？`, `！`, `︰`) inside Vietnamese prose to ASCII equivalents; normalized `.....` -> `...`; fixed spaces before/after punctuation while preserving decimal/thousands numbers (`1.000`, `1,5`, `0.5g`).
- **Chinese**: normalized `︰` -> `：`, `﹗` -> `！`, `‧` -> `·`, `.....` -> `……`.

### 2.8 Preserve proper nouns, era names, numbers, and measurement units
- Preserved all proper names (Tuệ Tĩnh, Nguyễn Bá Tĩnh, Phó Thanh Chủ, Lê Trần Đức, 许济群, 天下无疾, 賴玉瓚, etc.), era names/dates (`壬辰年`, `Cảnh Hưng`, `1061`, etc.), numbers (`37°C`, `1~2次`, `1.5g`, `49 hạt`, etc.), and traditional/modern measurement units (`thăng`, `đồng cân`, `lạng`, `phân`, `chỉ`, `g`, `kg`, `ml`, `兩`, `錢`, `分`, `升`, `合`, `枚`, `丸`).

### 2.9 Skip tables and images/figures
- **`nam-duoc-than-hieu-pdf.txt`**: skipped herb illustration plate pages (`Bảng II`, `Bảng IV`, `Bảng VI`, `Bảng VII` and their picture captions; 25 lines) and temple photo caption.
- **`phuong-te-hoc...`**: skipped embedded comparison tables (`表1四逆汤和通脉四逆汤鉴别表`, `表2小承气汤与厚朴三物汤鉴别表`, `表3抵当汤和抵当丸鉴别表`, `表4麻黄杏仁甘草石膏汤与麻黄汤鉴别表`, `表5历代衡量与辩的对照表`) and the 10-page table appendix (`附录1.常用中成药简表`).

---

## 3. Concrete Before / After Examples

| Category | File | Before | After |
|---|---|---|---|
| LaTeX & inline footnote | `nam-duoc-than-hieu-pdf.txt` | `-Hoa Kinh giới 2 đồng cân (đồng = đ/c$ ^{(2)} $).` | `-Hoa Kinh giới 2 đồng cân (đồng = đ/c).` |
| LaTeX italics | `nam-duoc-than-hieu-pdf.txt` | `Vỏ cây $ \textit{hòe} $ bỏ vỏ thở bên ngoài` | `Vỏ cây hòe bỏ vỏ thở bên ngoài` |
| Han-Nom + footnote + guillemets | `nam_khoa_nu_khoa.txt` | `thế là bởi «thủy hỏa hữu dư» chứ không phải «bất túc»... «ký tế» (2).` | `thế là bởi "thủy hỏa hữu dư" chứ không phải "bất túc"... "ký tế".` |
| Han-Nom + OCR pipe | `nam_khoa_nu_khoa.txt` | `\|富歸\|ĐƯƠNG QUY\|2 dòng` | `ĐƯƠNG QUY 2 dòng` |
| Marginalia & variant punct | `Pho_Thanh_Chu_Nu_Khoa.txt` | `身瘦不孕（29）` / `誰知是血虛之故乎﹗` / `眉批︰...` | `身瘦不孕` / `誰知是血虛之故乎！` *(marginalia removed)* |
| LaTeX temp & bullet | `Linh khởi điểm...` | `体温是$37^{\circ}C$...思考§中医认识人体` | `体温是37°C...思考：中医认识人体` |
| Emphasis dots & citation sup | `phuong-te-hoc...` | `$\underset{\cdot}{影}\underset{\cdot}{响}$` / `故曰备急”。$^{1}$` | `影响` / `故曰备急”。` |
| Woodblock folio & Bopomofo | `Tân_san_Nam_dược..._q10` | `六四末酒調服...每服一ㄐ` | `末酒調服...每服一` *(merged across page break)* |

---

## 4. Verification Audit

Post-processing automated verification across all 10 files in `./refined_text` confirmed:
- Han-Nom characters in Vietnamese-dominant files (`nam-duoc-than-hieu-pdf.txt`, `nam_khoa_nu_khoa.txt`): **0**
- Vietnamese diacritic characters in Chinese/Han-Nom-dominant files (all 8 files): **0**
- Remaining LaTeX/garbage symbols (`$`, `\`, `{`, `}`, `^`): **0**
- Remaining watermark URLs (`vnras.com`, `bbs.zhong-yao.net`): **0**
- Remaining standalone page numbers: **0**
- Remaining inline footnote markers (`(n)`, `[n]`): **0**
- Remaining Bopomofo/Kana OCR garbage: **0**
- Remaining multiple consecutive spaces or tabs: **0**
