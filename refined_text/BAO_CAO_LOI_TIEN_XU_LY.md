# Báo cáo rà soát lỗi tiền xử lý văn bản Hán–Việt

**Dự án:** NLP-midterm — corpus Y học cổ truyền Hán–Việt  
**Thành phần được rà soát:** `refined_text.zip/code/clean_texts.py`  
**Ngày rà soát:** 24/11/2026  
**Phạm vi:** Rà soát mã nguồn, báo cáo thống kê đi kèm và kiểm thử cục bộ. **Không** phải báo cáo đánh giá toàn bộ 10 TXT đã xử lý.

## 1. Căn cứ và nguyên tắc đánh giá

- **Tài liệu chuẩn ưu tiên:** `Hướng Dẫn Dóng Hàng câu Dịch nghĩa Hán - Việt.pdf`, nhất là mục **5 — Bước 2 (Làm sạch text)** và **Bước 3 (Tách câu)**.
- **Nguồn code:** `refined_text.zip/code/clean_texts.py` (991 dòng).
- **Tài liệu từ tác giả script:** `report/REFINEMENT_REPORT.md`, `report/refinement_stats.json` nằm trong ZIP.
- **Dữ liệu kiểm chứng:** `vi_test.txt` (đoạn Phó Thanh Chủ Nữ Khoa), một số input ngắn tạo riêng để thử hàm; các dữ liệu này **không** đại diện cho toàn bộ corpus.
- **Rule bổ sung của nhóm:** bảo toàn nội dung; lưu log **từng chỉnh sửa**, kèm văn bản trước/sau; ghi **4 ký tự trước + 4 ký tự sau** ở mỗi vị trí tách câu; không tự sửa nghĩa của bản dịch Quốc ngữ có sẵn; chỉ ghi file có chỉnh sửa vào output Kaggle.

**Cách đọc trạng thái:**
- **ĐÃ TÁI HIỆN:** Lỗi được xác nhận bằng cách chạy hàm trong ZIP trên đoạn thử.
- **RỦI RO TỪ CODE:** Đường xử lý có thể gây sai/mất thông tin nhưng chưa chứng minh đã xảy ra trong toàn bộ 10 sách.

### Tóm tắt mức ưu tiên

| ID | Mức độ | Trạng thái | Vấn đề |
|---|---|---|---|
| P01 | **P0 — nghiêm trọng** | ĐÃ TÁI HIỆN | Xóa nhầm dòng liều lượng khi hai vị thuốc có cùng số lượng |
| P02 | **P0 — nghiêm trọng** | ĐÃ TÁI HIỆN | Xóa nhầm câu gần giống nhưng **khác con số** |
| P03 | **P0 — nghiêm trọng** | ĐÃ TÁI HIỆN | Xóa ký tự OCR chưa rõ ở vị trí có thể là đơn vị đo, không giữ `[MASK]` |
| P04 | **P1 — cao** | ĐÃ TÁI HIỆN | Regex xóa `(1)`, `(3)` mà không xác minh đó có phải chú thích hay không |
| P05 | **P1 — cao** | ĐÃ TÁI HIỆN trên ví dụ tổng hợp | Ghép trang/chương không có dấu kết câu vào cùng một chuỗi |
| P06 | **P1 — cao** | THIẾU CHỨC NĂNG | Không có nhật ký **từng** thay đổi và vị trí nguồn |
| P07 | **P1 — cao** | THIẾU CHỨC NĂNG | Chưa tách câu; thiếu log ranh giới 4+4 ký tự |
| P08 | **P1 — cao** | RỦI RO TỪ CODE | Xóa nội dung hàng loạt theo mốc trang, chú thích hoặc regex cố định |
| P09 | **P1 — cao** | RỦI RO TỪ CODE | Xóa toàn bộ ký tự Hán trong bản Việt / từ Việt có dấu trong bản Hán mà không kiểm tra ngữ cảnh |
| P10 | P2 — trung bình | RỦI RO TỪ CODE | Ghép dòng/trang bằng heuristic có thể mất ranh giới đoạn hoặc phương thuốc |
| P11 | P2 — trung bình | THIẾU CHỨC NĂNG | Xuất mọi file, kể cả file không chỉnh sửa; không có kiểm tra đầu ra độc lập |

---

## 2. Lỗi đã tái hiện bằng kiểm thử

### P01 — Xóa nhầm liều lượng của một vị thuốc khác nhau

- **Vị trí code:** `deduplicate_adjacent_lines()`, **dòng 187–213**; được gọi từ `process_nam_khoa_nu_khoa()`, **dòng 488–489**.
- **Nguyên nhân:** So sánh từng dòng với tối đa ba dòng gần nhất, bỏ dòng giống hệt có độ dài từ sáu ký tự, **không xem xét dòng tên vị thuốc đứng trước**.
- **Minh chứng trên `vi_test.txt`:**

**Trước:**

```text
THỰC ĐỊA (3)
3 đồng cản.
ĐƠN BÌ
3 đồng cản.
BẠCH LINH
```

**Sau khi chạy `process_nam_khoa_nu_khoa()`:**

```text
THỰC ĐỊA
3 đồng cản.
ĐƠN BÌ
BẠCH LINH
```

Cụ thể, số lần xuất hiện `3 đồng cản.` giảm từ **2 xuống 1**. Dòng liều lượng của `ĐƠN BÌ` bị mất. Việc bỏ `(3)` bên cạnh `THỰC ĐỊA` là thao tác **khác**, cần kiểm tra theo P04.

- **Tác động:** Mất một đơn vị liều lượng; có thể gây hiểu sai toa thuốc và gây sai lệch so với bản Hán khi dóng hàng.
- **Đề xuất:** Không tự động deduplicate dòng ngắn chứa số/đơn vị; chỉ xóa khi xác định được hai **vùng OCR chồng lặp** thuộc cùng một vị trí trang. Ghi đề xuất xóa vào danh sách cần duyệt.

### P02 — Xóa nhầm hai câu gần giống nhưng khác con số

- **Vị trí code:** `deduplicate_adjacent_lines()`, **dòng 204–208**.
- **Nguyên nhân:** `difflib.SequenceMatcher(...).ratio() >= 0.92` có thể coi hai câu khác **một chữ số** là bản trùng.
- **Minh chứng (input tổng hợp):**

```text
Một thang thuốc có liều lượng 5 đồng cân.
Một thang thuốc có liều lượng 3 đồng cân.
```

**Kết quả thực thi:** Chỉ giữ dòng thứ nhất, xóa dòng thứ hai.

- **Tác động:** Vi phạm yêu cầu **giữ con số và đơn vị đo**; một thay đổi nhỏ về số có thể là khác biệt có ý nghĩa y học.
- **Đề xuất:** Không dùng ngưỡng tương đồng làm điều kiện đủ để xóa. So khớp chính xác các token số, đơn vị, tên thuốc, tên người; nếu có bất kỳ khác biệt nào, chỉ gắn nhãn `POSSIBLE_DUPLICATE`.

### P03 — Xóa ký tự OCR chưa xác định có thể đại diện cho đơn vị thuốc

- **Vị trí code:** `process_tan_san_nam_duoc()`, **dòng 897–908**, đặc biệt `re.subn(r"[\u3100-\u312f\u3040-\u30ff]", "", p)`.
- **Nguyên nhân:** Xóa thẳng ký tự Bopomofo/Kana được OCR nhận sai, dù phần chú thích trong code thừa nhận nó **có thể** thay cho `錢`, `寸`, `分` hoặc một chữ khác.
- **Minh chứng (input tổng hợp):**

| Trước | Sau | Nhận xét |
|---|---|---|
| `每服一ㄐ` | `每服一` | Mất ký tự sát số lượng; đơn vị còn thiếu |
| `每服二ㄔ` | `每服二` | Tương tự |

- **Tác động:** Mất dấu vết OCR để đối chiếu ảnh gốc và có thể làm khuyết đơn vị đo.
- **Đề xuất:** Giữ ký tự gốc trong bản OCR thô, đưa vào review; với trường hợp **không đọc được** khi xây dựng dữ liệu, dùng `[MASK]` theo hướng dẫn, hoặc khôi phục từ **bản khác có dẫn nguồn**. Không tùy tiện đoán là `錢`.

### P04 — Xóa chỉ mục dạng `(n)` khi chưa đủ căn cứ phân biệt chú thích và nội dung

- **Vị trí code:** `remove_inline_footnote_markers_vn()`, **dòng 168–184**; cùng phần xử lý chú thích dạng LaTeX, **dòng 79–90**.
- **Minh chứng (input tổng hợp):**

```text
Phương thuốc (1) này gọi là: Phục linh (3) đồng cân; dùng 2 (3) lần.
```

Sau xử lý:

```text
Phương thuốc này gọi là: Phục linh đồng cân; dùng 2 (3) lần.
```

- **Đánh giá:** Hướng dẫn **có yêu cầu bỏ chỉ mục chú thích** như `(1)` và `[1]`, nên không phải mọi lần xóa đều sai. Lỗi nằm ở chỗ quy tắc xóa **chưa xác minh vai trò của từng dấu ngoặc**; nếu `(3)` là liều lượng hoặc phần nội dung hợp lệ, mất nó sẽ làm thay đổi nghĩa.
- **Đề xuất:** Chỉ tự xóa các chỉ mục đã nhận diện chắc chắn là chú thích; số/đơn vị trong phần thân toa thuốc phải được bảo vệ. Những trường hợp không chắc chắn ghi `POSSIBLE_FOOTNOTE`.

### P05 — Ghép ranh giới hai chương khi không có dấu câu cuối trang

- **Vị trí code:** `merge_cjk_pages()`, **dòng 631–645**.
- **Nguyên nhân:** Nếu trang trước không kết thúc bằng một số dấu câu, script ghép trực tiếp (`prev + p`), không kiểm tra phần sau có phải tiêu đề mục mới.
- **Minh chứng (input tổng hợp):**

```python
merge_cjk_pages(["第一章论述法", "第二章论述法"], stats)
```

Cho ra:

```text
第一章论述法第二章论述法
```

- **Tác động:** Mất ranh giới mục/chương; nguy cơ làm hỏng phân đoạn nguồn và dóng hàng.
- **Đề xuất:** Phát hiện tiêu đề, chương, mục và dấu chuyển trang trước khi ghép; nếu chưa chắc chắn, giữ ranh giới hoặc đánh dấu cần kiểm tra.

---

## 3. Những rủi ro và phần thiếu chức năng

### P06 — Thiếu audit log cho từng chỉnh sửa

- **Vị trí:** Hầu hết hàm chỉ tăng bộ đếm `stats`; `main()` ghi `refinement_stats.json` ở **dòng 981–986**.
- **Hiện trạng:** Có thể biết *đã xóa bao nhiêu* nhưng không xác định chính xác **xóa đoạn gì, trước/sau ra sao, ở dòng/ký tự nào, do rule nào**. Các số liệu thống kê không thay thế nhật ký kiểm toán.
- **Cần bổ sung:** `change_id`, `file`, `rule_id`, `original_line`, `original_start/end`, `before`, `after`, `reason`, `review_status`; cả thao tác xóa, thay ký tự, ghép dòng/trang cũng phải có bằng chứng và vùng nguồn.

### P07 — Chưa tách câu và chưa ghi ngữ cảnh tách 4+4

- **Vị trí:** `main()` chỉ gọi hàm làm sạch rồi ghi TXT, không có bước sentence segmentation.
- **Hiện trạng:** Không tạo `han_sentences.txt`, `vi_sentences.txt` hay JSONL có offset; không có `segmentation_changes.txt`.
- **Cần bổ sung:** Tách câu theo gợi ý `。！？；` với Hán và `.?!;:` với Việt, kết hợp nghĩa và cấu trúc; log cho mỗi điểm tách gồm **4 ký tự Unicode ngay trước + 4 ký tự ngay sau**, vị trí và lý do. Tránh cắt số thập phân, tên viết tắt, thứ tự, danh sách vị thuốc.
- **Lưu ý:** Đây là phần thiếu so với **pipeline tiền xử lý/tách câu của nhóm**, không phải lỗi mất dữ liệu trong riêng nhiệm vụ làm sạch.

### P08 — Lọc/xóa nội dung theo mốc cứng có thể cắt nhầm

**Các nhánh đáng kiểm tra:**

- `process_nam_khoa_nu_khoa()` **dòng 449–462**: cắt toàn bộ phần sau dòng `附註女科`.
- `process_nam_duoc_than_hieu()` **dòng 275–292**: lấy mốc `QUYỂN ĐẦU` và `MỤC LỤC`, dựa vào bản nguồn cụ thể.
- `process_linh_khoi_diem()` **dòng 657–674**: cắt từ `体验——中医入门之道` đến trước `附录C`.
- `process_phuong_te_hoc()` **dòng 736–750**: dừng ở `附录1` và bỏ các trang mục lục theo dấu hiệu hình thức.
- `process_phuong_te_hoc()` **dòng 754–774**: regex xóa từ nhãn bảng đến cuối trang; có thể cuốn theo văn xuôi nếu phần cần giữ nằm sau bảng.
- `process_pho_thanh_chu()` **dòng 562–572**: loại khối `眉批` đến dòng trống hoặc `歌括`; cần xác minh ranh giới ghi chú.

**Trạng thái:** Chưa có dữ liệu TXT đầy đủ để chứng minh các quy tắc này đã cắt nhầm phần thân sách. Đây là các **nguy cơ cần kiểm tra dựa trên diff** và vùng trang gốc, không nên ghi thành lỗi thực tế của toàn corpus.

### P09 — Lọc script chéo quá mạnh, khó truy vết

- **Vị trí:** `strip_han_nom_in_vn()` **dòng 50–63**; `strip_vietnamese_in_cjk()` **dòng 66–72**.
- **Rủi ro:** Xóa tên thuốc hoặc tên tác phẩm bằng chữ Hán lẫn trong bản Việt; xóa chú giải tiếng Việt có dấu ở bản Hán. Với workflow của nhóm, xóa chữ Hán khỏi bản Việt có thể là mục tiêu chủ ý, **nhưng vẫn phải biết đoạn bị xóa là gì** và không để lẫn khoảng trắng bất thường.
- **Đề xuất:** Lưu phần bị xóa vào audit và cho phép xem lại trong bản nguồn. Với dữ liệu đôi ngữ, đừng dùng việc xóa chữ Hán để che lấp lỗi dịch.

### P10 — Ghép trang/dòng có thể phá cấu trúc ngữ nghĩa

- **Vị trí:** `process_nam_duoc_than_hieu()` **dòng 372–424**; `process_nam_khoa_nu_khoa()` **dòng 491–518**; `merge_cjk_pages()` **dòng 631–645**; `process_tan_san_nam_duoc()` **dòng 915–938**.
- **Rủi ro:** Dựa nhiều vào độ dài dòng, chữ hoa/chữ thường, dấu câu cuối dòng và tiền tố toa thuốc. Những quy tắc này có thể ghép công thức thuốc, tiêu đề hoặc hai đơn vị ngữ nghĩa khác nhau.
- **Đề xuất:** Bảo vệ vùng tiêu đề, vị thuốc, liều lượng; ghi nguồn của từng đoạn sau ghép; đưa các trường hợp nhập nhằng sang review thay vì ghép tự động.

### P11 — Xuất mọi file và chỉ báo thống kê tổng hợp

- **Vị trí:** `main()` **dòng 978–986**.
- **Hiện trạng:** Ghi TXT đầu ra cho **mọi** file được xử lý, không có kiểm tra `refined != raw`. Báo cáo chỉ có bộ đếm, không có diff hay tỉ lệ giữ nội dung theo từng vùng nguồn.
- **Đề xuất:** Chỉ lưu file có nội dung thực sự đổi, luôn giữ nguyên input; xuất `changes.txt`/`review.jsonl` và summary các file thực sự thay đổi; có xác thực tính đầy đủ nguồn.

---

## 4. Kiểm tra báo cáo số liệu trong ZIP

`refinement_stats.json` báo cáo xử lý 10 file, chẳng hạn:

| File | Input → output (ký tự) | Chỉ số tác giả ghi nhận | Nhận xét kiểm toán |
|---|---:|---|---|
| `nam_khoa_nu_khoa.txt` | 258.055 → 216.036 | 1 dòng trùng bị xóa; 399 chỉ mục chú thích; 1.398 nhóm ký tự Hán bị xóa | Bộ đếm không xác nhận nội dung bị xóa là hợp lệ |
| `Pho_Thanh_Chu_Nu_Khoa.txt` | 54.204 → 53.295 | 2 dòng trùng, 21 dòng chú thích | Cần so từng vùng có xóa/ghép |
| `nam-duoc-than-hieu-pdf.txt` | 1.051.267 → 1.002.115 | 5 dòng trùng, 125 dòng chú thích, 36 lần ghép | Có nguy cơ sai ở số/liều lượng và dòng bị ghép |
| Các tập `Tân san Nam dược ...` | Nhiều sách | Hàng chục đến hơn 100 lần ghép trang/tập | Cần kiểm tra ranh giới toa và các ký tự bị xóa |

**Giới hạn kiểm chứng:** Trong `refined_text.zip` chỉ có code và file báo cáo, **không có 10 TXT đầu vào/đầu ra** để đối chiếu đầy đủ. Các số liệu trong bảng trên là **báo cáo do script/tác giả cung cấp**, không phải các kết quả được tái chạy độc lập trên cả 10 sách. Những số như “0 ký tự Hán còn lại” hay “0 số trang còn lại” không chứng minh văn bản đã bảo toàn đúng nội dung.

## 5. Hướng khắc phục và tiêu chí nghiệm thu

### P0 — Phải sửa trước khi dùng dữ liệu để dịch/dóng hàng

1. **Tắt auto-delete dòng gần trùng** ở các vùng văn xuôi, toa thuốc và liều lượng; phát hiện và đưa vào review.
2. **Bảo vệ số và đơn vị đo** trong mọi phép xóa, thay, ghép; không bỏ số chỉ vì câu gần giống.
3. **Không tự xóa ký tự OCR không xác định ở vị trí có nghĩa**; dùng `[MASK]` theo quy định khi thật sự không đọc được, lưu chuỗi gốc và bằng chứng.
4. Tạo bộ test hồi quy bắt buộc gồm các lỗi P01–P05.

### P1 — Phải có trước khi công nhận corpus sạch

5. Xuất **log từng chỉnh sửa** kèm trước/sau, tọa độ, nguyên nhân và trạng thái review.
6. Kiểm tra thủ công/LLM hỗ trợ các vùng bị cắt theo mốc cứng, đặc biệt chú thích, tiêu đề và bảng.
7. Tách câu Hán/Việt theo nghĩa, xuất ID/offset và **log 4+4** cho mọi ranh giới tách.
8. Kiểm tra đủ nội dung nguồn, tỉ lệ vùng chưa giải quyết và danh sách các thay đổi có rủi ro.

### P2 — Tích hợp vận hành

9. Chuyển thành module import được, có tham số paths và **chạy được bằng Kaggle Save Version từ đầu**.
10. Chỉ ghi **file có chỉnh sửa** vào `/kaggle/working`, không sao chép TXT hoặc mã nguồn đầu vào.
11. Kiểm tra output đối chiếu với nguồn (không tự động ghi PASS khi thiếu input gốc).

### Bộ kiểm thử chấp nhận tối thiểu

| Test | Tiêu chí đạt |
|---|---|
| Hai vị thuốc có cùng liều lượng | Giữ **cả hai** dòng liều lượng |
| Hai câu chỉ khác số `5`/`3` | Giữ **cả hai** |
| `每服一ㄐ` | Không biến âm thầm thành `每服一`; giữ dấu vết và đánh dấu cần kiểm tra |
| `(1)`, `(3)` trong công thức thuốc | Không xóa khi chưa phân loại chắc chắn |
| Hai tiêu đề CJK liên tiếp | Không dính thành một chuỗi |
| Ghép từ bị ngắt trang | Vẫn khôi phục được vị trí và nội dung gốc |
| Tách câu | Mỗi split có offset và đủ ngữ cảnh 4+4 khi tồn tại |
| File không sửa | Không xuất bản sao vào output |

## 6. Kết luận

`clean_texts.py` có ưu điểm là các quy tắc **đặc thù cho từng quyển sách** và làm sạch nhiều dạng OCR. Tuy nhiên, hiện có **ít nhất ba nhóm lỗi mất dữ liệu đã tái hiện** (dòng liều lượng, số trong câu gần trùng, ký tự OCR chưa rõ), cùng các thiếu sót về audit và tách câu.

**Khuyến nghị:** Không bỏ toàn bộ các quy tắc đã viết, nhưng **chưa sử dụng output làm corpus chính thức**. Ưu tiên sửa P01–P03, khóa các thao tác xóa không chắc chắn, bổ sung audit và chạy so sánh trước/sau trên TXT thật. Những cảnh báo còn lại cần xác minh trước khi kết luận đã gây sai trong toàn bộ sách.

> Báo cáo này chỉ đánh giá **tiền xử lý**. Không trộn các lỗi dịch của LLM hoặc lỗi dóng hàng Bertalign/Vecalign/API vào cùng danh sách lỗi tiền xử lý.
