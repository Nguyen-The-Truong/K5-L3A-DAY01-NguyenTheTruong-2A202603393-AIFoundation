# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603393
- **Họ và tên:** Nguyen The Truong
- **SEED** (in ở ô đầu notebook): 4114

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: 15/15 đạt. Đủ 10 dự đoán `predict(...)`.

## 2. Kết quả chính

LLM chỉ chạy trên SUB (40 tin), nên so chất lượng với LLM bằng cột F1 (SUB 40).

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 0.910 | 0.889 | 0.41 | 0 |
| 2. BoW + MLP | 0.944 | 0.919 | 0.17 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.910 | 0.976 | 45.3 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.964 | 0.950 | 30.5 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.964 | 0.974 | 35.1 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | chỉ SUB | 0.864 | 2743.6 | 0.005 |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | chỉ SUB | 0.976 | 2443.0 | 0.010 |
| 4c. LLM nhỏ few-shot | chỉ SUB | 0.927 | 3276.7 | 0.013 |

Ghi chú: hàng 3a có F1 SUB cao (0.976) nhưng F1 test chỉ 0.910 (precision 0.862), vì SUB cân bằng 50/50 nên che việc kNN chặn nhầm nhiều tin ham. Chênh dưới ~0.05 trên SUB (40 tin) coi như hoà.

## 3. Dự đoán và thực tế

1. **`ml_lr`** — Đoán: lr=0.001 thì loss giảm rất chậm, lr=50 thì loss dao động mạnh, không giảm đều. Thực tế: lr=0.001 đúng (loss 0.693 → 0.627), nhưng lr=50 loss vọt lên khoảng 2.6 ở vài vòng đầu rồi tụt về gần 0 (loss cuối 0.007). Lệch vì bước cập nhật tỉ lệ với `error = p − y`, mà `error` luôn nằm trong [−1, 1] và tự co về 0 khi mô hình đã đoán đúng, nên không dao động mãi. Loss train thấp ở lr=50 chưa chắc tốt trên test (có thể học vẹt).
2. **`emb_tricky`** — Đoán: Encoder+LogReg đúng nhiều hơn BoW trên 6 tin "ngụy trang". Thực tế: hoà 5/6. Lệch vì BoW vẫn bắt được tin spam có các từ như `your`, `mobile`, `reply` (nằm trong danh sách từ đẩy mạnh về spam), và cả hai cùng trượt tin phishing "Last chance! Your account rewards expire tonight...". 6 tin là quá ít để kết luận, một tin sai đổi tỉ lệ 17%.
3. **`hard_eval`** — Đoán: LLM khó nhất ở nhóm prompt injection. Thực tế: LLM làm tốt nhóm đó (83–100%), khó nhất lại ở nhóm tưởng spam nhưng không phải (70–75%) và nói về spam (75–80%). Lệch vì LLM đề phòng quá mức khi thấy từ khoá mạnh (FREE, Cash prize, won £1000) dù ngữ cảnh là tin bạn bè hoặc kể chuyện.

## 4. Phân tích 50 case khó

Mô hình tụt nhiều nhất là **BoW + MLP**: F1 từ 0.919 (test thường, SUB) xuống khoảng 0.32 trên bộ case khó, accuracy chỉ còn khoảng 0.52–0.54 vì gần như đoán ham cho mọi tin. BoW + LogReg cũng tụt từ 0.889 xuống khoảng 0.31. BoW trượt nặng ở nhóm né từ khoá (0–17%), spam giọng tự nhiên (8–10%) và thao túng (12%). Điểm trên tập test thường quá lạc quan vì nó cùng phân phối với tập train (chỉ bỏ trùng tuyệt đối, spam mẫu chỉ khác số điện thoại hay mã thưởng vẫn còn ở cả hai tập), nên mô hình chỉ cần nhớ từ khoá là đạt điểm cao.

Hai case cụ thể:

- **#47 [Nói về spam], thật = ham:** "lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th..." LLM (4d) đoán spam. Cơ chế: LLM thấy cụm từ khoá lừa đảo mạnh (`won £1000`, `call`, số 0906) và đề phòng quá mức, không phân biệt được tin **kể về** một tin lừa đảo với chính tin lừa đảo.
- **Tin phishing trong thử thách của tôi** "Your parcel delivery has been suspended due to an incomplete address. Confirm details here immediately." (thật = spam): BoW đoán ham. Cơ chế: tin viết trang trọng, không có từ khoá spam mạnh (free, call, txt, win), các từ phổ biến như `your`, `has`, `been` không đủ đẩy về spam. Encoder + LogReg cũng đoán ham, có thể vì giọng văn giống thông báo hợp lệ của hãng chuyển phát.

Prompt phòng thủ (4d) giúp nhóm prompt injection từ 83% (4a) lên 100% và không làm hỏng nhóm nào (tổng 88% → 92% trên 50 case gốc).

## 5. Thử thách cá nhân

**Đề #5 (Embedding):** viết 4 tin của riêng mình (2 spam né từ khoá, 2 ham chứa `free`/`win`), so dự đoán BoW và Encoder + LogReg, giải thích từng tin.

| STT | Tin | Nhãn | BoW | Encoder + LogReg |
|---|---|---|---|---|
| 1 | Your parcel delivery has been suspended due to an incomplete address. Confirm details here immediately. | spam | ham ✗ | ham ✗ |
| 2 | Unusual activity detected on your bank profile. Verify your credentials now to prevent permanent account suspension. | spam | ham ✗ | ham ✗ |
| 3 | Hey, are you free this weekend? Let's go grab some coffee and catch up. | ham | ham ✓ | ham ✓ |
| 4 | Did our school team win the soccer match yesterday? Let me know the score! | ham | ham ✓ | ham ✓ |

Cả hai mô hình đúng 2/4, cùng trượt hai tin spam. **Giải thích:** hai tin spam viết trang trọng, không chứa từ khoá spam mạnh nên BoW không thấy gì để bám; với Encoder, giọng văn giống thông báo hợp lệ của hãng chuyển phát hay ngân hàng nên vector nằm gần vùng tin ham. Hai tin ham có `free` và `win` được đoán đúng vì chữ khoá đó bị át bởi các từ đời thường kéo về ham (hey, coffee, team, know, me), và với Encoder thì nghĩa cả câu là rủ đi cà phê hay hỏi kết quả trận đấu. Kết luận: đổi từ BoW sang Encoder không cứu được loại spam giả danh tổ chức, kiểu mà LLM làm tốt hơn trong stress test (nhóm thao túng và spam giọng tự nhiên đều 100%).

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (khoảng 116 tin/giây trung bình), p95 < 50 ms/tin.

- **Đạt độ trễ:** BoW + LogReg (0.41 ms), BoW + MLP (0.17 ms), Encoder + kNN (45 ms), Encoder + LogReg (30.5 ms), Encoder + MLP head (35 ms). Cả ba LLM đều không đạt (p95 2,4–3,3 giây).
- **Đạt cả độ trễ lẫn chất lượng:** Encoder + LogReg và Encoder + MLP head (F1 test 0.964).
- **Tôi chọn Encoder + LogReg (3b):** F1 test 0.964, p95 30.5 ms, chi phí API bằng 0. Đánh đổi: thông lượng khoảng 66 tin/s mỗi máy CPU nên cần ít nhất 2 máy (thêm dư cho giờ cao điểm), và nó vẫn trượt spam giả danh trang trọng (xem mục 5).
- **Nếu dùng LLM:** $0.005–$0.013 mỗi 1000 tin, tương đương khoảng $50–$130 mỗi ngày, khoảng $1.500–$3.900 mỗi tháng, và vẫn vượt SLA gấp hơn 48 lần.
- **Đề xuất thêm (router):** BoW + MLP (p95 0.17 ms) xử lý các tin chắc chắn (p gần 0 hoặc gần 1); chỉ các tin lưỡng lự (p trong [0.3, 0.7]) chuyển cho LLM. Phần lớn tin được xử lý rẻ và nhanh, LLM chỉ tốn tiền ở các case khó mà BoW yếu.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 0–3 | Giải thích khái niệm (bag-of-words, sigmoid, gradient descent, cosine...), xem lại code của tôi | Giải thích khái niệm, chỉ lỗi (ví dụ dùng `split()` thay vì `tokenize`) | Code TODO 1.1 đến 3.3 do tôi tự viết, tự chạy ô kiểm tra |
| 0–3 | Nháp câu trả lời Câu hỏi 0.1, 1.B, Bắt lỗi 1–2, Checkpoint A–B, Câu hỏi 3 | Bản nháp câu trả lời | Viết lại bằng lời mình, đối chiếu số liệu của mình |
| 1–5 | Nháp các dự đoán `ml_lr`, `emb_tricky`, `llm_models`, `llm_fewshot`, `tradeoff`, `hard_eval` | Câu dự đoán | Đối chiếu với kết quả thật, nêu chỗ lệch |
| 4 | Viết code TODO 4.1 đến 4.4 | Code hoàn chỉnh | Đọc kỹ, chạy ô kiểm tra, chỉnh retry cho giới hạn Gemini, đổi sang Groq |
| 4–5 | Nháp Câu hỏi 4, 5, 6, Bắt lỗi 3, phản ánh cuối bài | Bản nháp câu trả lời | Viết lại, kiểm tra lại số liệu |
| 5 | Nháp 6 case `MY_CASES`, lời giải 5 phép tính tay | Case và lời giải có các bước | Tự chép lại bằng tay lên giấy, chạy ô mã băm |
| 5 | Thử thách cá nhân #5 | Chỉ giải thích khái niệm và chỉ lỗi tên biến, không viết code | Code và 4 tin do tôi tự viết, tự giải thích kết quả |

**Điều hiểu ra hôm nay:** điểm cao trên tập test thường chưa chứng minh mô hình "hiểu" spam (BoW tụt từ khoảng 0.92 xuống khoảng 0.32 ở stress test), và model lớn hơn hay prompt dài hơn không phải lúc nào cũng đáng tiền: few-shot của model nhỏ đắt hơn model lớn zero-shot mà vẫn kém hơn. Mỗi cách tiếp cận có loại lỗi riêng (BoW không thấy nghĩa, Encoder lẫn tin giả danh trang trọng, LLM đề phòng quá mức), nên chọn phương án phải dựa trên số đo và yêu cầu của sản phẩm chứ không dựa trên cảm giác.
