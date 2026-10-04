# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Hoàng Anh  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores Comparison

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 0.8167 | 0.8350 | +0.0183 |
| Answer Relevancy | 0.6873 | 0.6715 | -0.0158 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.7667 | -0.1583 |

---

## Bottom-5 Failures Analysis

Các câu hỏi thất bại nặng nhất được trích xuất chính xác theo thứ tự tăng dần của `avg_score` trong báo cáo thực tế `reports/ragas_report.json`:

### #1
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (Score: 0.0 | Avg RAGAS Score: 0.1458)
- **Error Tree:** 
  1. Output đúng? → Không ("Không tìm thấy.").
  2. Context đúng? → Không đủ dữ liệu trong RAGAS report để xác định chính xác retrieved context có chứa bảng lương hay không.
  3. Query OK? → Query chứa các khái niệm kết hợp ("lương thử việc" + "Junior mức cao nhất").
- **Observed behavior:** Model trả lời "Không tìm thấy.", dẫn tới điểm Faithfulness = 0.0 do không cung cấp được câu trả lời trùng khớp với Ground Truth.
- **Likely root cause (Hypothesis):** Giả thuyết là do Retrieval Failure. Thông tin bảng lương Junior (20 triệu) và quy định tỷ lệ lương thử việc (85%) nằm rải rác ở 2 tài liệu/chunk khác nhau. Hybrid search có thể đã không ghép đủ cả 2 chunks này vào top context.
- **Evidence needed:** Cần log trực tiếp mảng `contexts` được trả về từ `HybridSearch` + `CrossEncoderReranker` cho query này.
- **Suggested fix:** Áp dụng Multi-query Decomposition hoặc Parent-Child Chunking với kích thước context rộng hơn để bao phủ cả ngạch lương lẫn chính sách thử việc.

### #2
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (Score: 0.0 | Avg RAGAS Score: 0.2500)
- **Error Tree:**
  1. Output đúng? → Không ("Không tìm thấy.").
  2. Context đúng? → Không đủ dữ liệu trong RAGAS report để xác định retrieved context.
  3. Query OK? → Query dùng số liệu cụ thể ("55 triệu") thay vì cụm từ khoảng hạn mức trong văn bản ("trên 50.000.000 VNĐ").
- **Observed behavior:** Model trả về "Không tìm thấy.", khiến chỉ số Faithfulness bằng 0.0.
- **Likely root cause (Hypothesis):** Giả thuyết là do sự khác biệt giữa con số cụ thể trong câu hỏi ("55 triệu") và dạng điều kiện khoảng ("trên 50 triệu") trong tài liệu gốc khiến Dense Embedding & BM25 bỏ sót chunk điều kiện hạn mức.
- **Evidence needed:** Kiểm tra danh sách candidate chunks được BM25 và Vector Search trả về trước và sau khi Rerank.
- **Suggested fix:** Thêm bước Query Rewriting / Range Expansion để chuẩn hóa các con số cụ thể thành biểu thức khoảng trước khi thực hiện tìm kiếm.

### #3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior có 9 năm thâm niên sẽ được nghỉ tổng cộng 18 ngày phép năm (15 ngày phép cơ bản + 3 ngày phép thêm cho 9 năm thâm niên). Về lương, thông tin không được cung cấp trong context.
- **Worst metric:** Answer Relevancy (Score: 0.0 | Avg RAGAS Score: 0.6250)
- **Error Tree:**
  1. Output đúng? → Đúng 1 phần (tính chính xác 18 ngày phép, nhưng thiếu thông tin lương Senior).
  2. Context đúng? → RAGAS report không lưu context riêng của câu hỏi này nên chưa khẳng định được context thiếu hay LLM bỏ qua.
  3. Query OK? → Đây là câu hỏi phức hợp (multi-part query) đòi hỏi truy xuất thông tin từ 2 chủ đề độc lập.
- **Observed behavior:** Answer Relevancy bằng 0.0 do câu trả lời chưa giải quyết vế thứ hai của câu hỏi (khung lương Senior).
- **Likely root cause (Hypothesis):** Dạng câu hỏi Multi-hop ghép 2 vấn đề (Phép năm & Khung lương). Retrieval có thể đã bị thiên vị nghiêng về các chunk nghỉ phép và bỏ sót chunk ngạch lương P3-P4.
- **Evidence needed:** Kiểm tra mảng `contexts` xem có chứa văn bản Quy chế tiền lương ngạch Senior hay không.
- **Suggested fix:** Tách câu hỏi multi-part thành 2 sub-queries: (1) "Số ngày phép của Senior 9 năm thâm niên" và (2) "Khung lương ngạch Senior P3-P4", sau đó gộp kết quả tìm kiếm.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Cần phê duyệt của Giám đốc điều hành (CEO).
- **Worst metric:** Context Precision (Score: 0.3333 | Avg RAGAS Score: 0.6541)
- **Error Tree:**
  1. Output đúng? → Đúng ý chính (CEO phê duyệt), nhưng không đề cập lưu ý phụ về bảo hiểm.
  2. Context đúng? → Context Precision thấp (0.3333) phản ánh có sự xuất hiện của chunk nhiễu trong top context.
  3. Query OK? → Query rõ ràng.
- **Observed behavior:** Model trả lời đúng cấp phê duyệt (CEO). Điểm yếu nhất rơi vào Context Precision (0.3333).
- **Likely root cause (Hypothesis):** RAGAS report không lưu vị trí ranking cụ thể của từng chunk. Giả thuyết là trong top context được xếp hạng có chứa các chunk quy trình nghỉ phép chung đứng ở vị trí đầu, đẩy chunk chi tiết quy định nghỉ không lương 16-30 ngày xuống rank thấp hơn, khiến Context Precision sụt giảm.
- **Evidence needed:** Kiểm tra thứ tự xếp hạng (ranking order) thực tế và điểm tương quan của các chunk trong mảng `contexts`.
- **Suggested fix:** Tăng hệ số phạt nhiễu trong Reranker hoặc điều chỉnh trọng số RRF để ưu tiên các chunk có độ khớp chính xác với khoảng số ngày nghỉ phép.

### #5
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Không tìm thấy.
- **Worst metric:** Answer Relevancy (Score: 0.0 | Avg RAGAS Score: 0.7500)
- **Error Tree:**
  1. Output đúng? → Trả lời "Không tìm thấy." thay vì trả lời trực tiếp "Không".
  2. Context đúng? → RAGAS report không lưu trữ nội dung context thô cho câu hỏi này.
  3. Query OK? → Query trực tiếp, rõ ràng.
- **Observed behavior:** Model trả về "Không tìm thấy.", dẫn tới Answer Relevancy = 0.0.
- **Likely root cause (Hypothesis):** Giữ ở mức giả thuyết vì chưa có bằng chứng khẳng định System Prompt hay Retrieval gây ra lỗi. Giả thuyết 1: Retrieval không lấy được chunk chứa quy định PVI. Giả thuyết 2: Context retrieved có chứa quy định PVI (chỉ dành cho nhân viên ký HĐLĐ chính thức), nhưng System Prompt chỉ thị quá khắt khe ("Trả lời CHỈ dựa trên context") khiến LLM không tự thực hiện phép suy luận phủ định gián tiếp ("Thử việc != HĐLĐ chính thức").
- **Evidence needed:** Inspection trực tiếp mảng `contexts` được retrieved thực tế lẫn prompt được gửi tới LLM cho câu hỏi này.
- **Suggested fix:** Tinh chỉnh System Prompt cho phép LLM thực hiện phép suy luận hợp logic từ điều kiện loại trừ gián tiếp trong văn bản.

---

## Case Study Deep-Dive (Presentation Case)

**Câu hỏi chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Phân tích theo Error Tree Framework:**
1. **Output kiểm tra (Kết luận chắc chắn):** Model trả lời đúng phần tính toán số ngày phép năm (18 ngày), nhưng trả lời thiếu phần khung lương Senior (*"Về lương, thông tin không được cung cấp trong context."*).
2. **Context kiểm tra (Thông tin chứng minh):** RAGAS report **không lưu trữ mảng `contexts` riêng của câu hỏi này**. Vì vậy, không thể suy ra chắc chắn chunk ngạch lương Senior bị thiếu chỉ từ điểm Context Recall aggregate của toàn bộ 20 câu hỏi. Ta chỉ ghi nhận thực tế câu trả lời bị thiếu thông tin ở vế thứ hai.
3. **Query Analysis:** Đây là dạng câu hỏi **Multi-part / Multi-hop**, hỏi đồng thời 2 thực thể thuộc 2 tài liệu quy định khác nhau (Quy chế nghỉ phép v2024 và Quy chế tiền lương).
4. **Giả thuyết & Đề xuất:** Giả thuyết là bước Retrieval bị thiên vị nghiêng về tài liệu nghỉ phép. Nếu có thêm 1 giờ, sẽ triển khai **Query Decomposition / Sub-query Generation** để tự động tách câu hỏi phức hợp thành các câu hỏi đơn độc lập và tăng `top_k` candidate context.

---

## Tổng kết Đánh giá Pipeline Production vs Naive Baseline

Dựa trên số liệu đối chiếu thực tế từ `reports/naive_baseline_report.json` và `reports/ragas_report.json`:

1. **Faithfulness:** Cải thiện nhẹ (+0.0183, từ 0.8167 lên 0.8350). Việc bổ sung Cross-Encoder Reranking (M3) và Context Enrichment (M5) giúp giảm bớt hiện tượng suy diễn sai của LLM.
2. **Answer Relevancy:** Giảm nhẹ (-0.0158, từ 0.6873 xuống 0.6715). Một số câu hỏi nhận được phản hồi "Không tìm thấy." thay vì câu trả lời trực tiếp.
3. **Context Precision:** Giảm nhẹ (-0.0083, từ 0.9250 xuống 0.9167). 
4. **Context Recall:** **Giảm đáng kể (-0.1583, từ 0.9250 xuống 0.7667)**. 

**Kết luận chung:**  
Hệ thống Production RAG hiện tại **chưa cải thiện toàn diện** so me với Naive Baseline. Việc áp dụng Hierarchical Chunking (M1) và Context Enrichment (M5) giúp nâng cao độ trung thực của câu trả lời (Faithfulness), nhưng làm sụt giảm khả năng bao phủ thông tin (Context Recall) đối với các câu hỏi phức hợp multi-hop và các truy vấn chứa con số cụ thể. 

Báo cáo Failure Analysis này đóng vai trò cơ sở để xác định chính xác các điểm nghẽn và lập kế hoạch nâng cấp cho các phiên bản tiếp theo.
