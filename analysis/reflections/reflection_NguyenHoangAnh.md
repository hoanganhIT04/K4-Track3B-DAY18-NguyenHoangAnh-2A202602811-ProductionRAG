# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Hoàng Anh  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là đối chiếu chi tiết giữa các khái niệm lý thuyết trong bài giảng và việc triển khai thực tế trong codebase Lab 18:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích thực tế |
|----------------|--------|-------------|----------------------------------|
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tách tài liệu thành các đoạn con (child ~300 chars) nằm trong đoạn cha (parent ~1200 chars). Giúp bảo toàn ngữ nghĩa tổng thể của đoạn văn bản dài mà vẫn tối ưu cho việc truy xuất vector ở độ phân giải nhỏ. |
| **Hybrid Search & RRF Fusion** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 (lexical search cho từ khóa chính xác như con số, thuật ngữ "PVI", "MFA") và Dense Search (Sentence-Transformers cho từ đồng nghĩa). RRF giúp dung hòa rank score giữa 2 phương pháp mà không cần chuẩn hóa điểm số khác biệt. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `cross-encoder/ms-marco-MiniLM-L-6-v2` để chấm điểm tương quan từng cặp (query, candidate context). Giúp loại bỏ các chunk nhiễu và đẩy chunk có độ liên quan ngữ nghĩa cao nhất lên Top-K trước khi đưa vào LLM. |
| **RAGAS 4 Metrics Evaluation** | M4 | `evaluate_ragas()` | Đánh giá tự động hệ thống RAG qua 4 chỉ số: Faithfulness (0.8350), Answer Relevancy (0.6715), Context Precision (0.9167), Context Recall (0.7667). Các metric giúp định lượng chất lượng pipeline và hỗ trợ xác định các failure case cần phân tích sâu hơn. |
| **Contextual Prepend Enrichment** | M5 | `enrich_chunks()`, `_enrich_single_call()` | Dùng GPT-4o-mini để tóm tắt context tổng thể và tạo tiêu đề bổ sung (prepend) vào từng chunk độc lập. Kỹ thuật này giúp giảm vấn đề "out-of-context" khi chunk đứng riêng lẻ mất đi ngữ cảnh tài liệu gốc. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi kỹ thuật gặp phải

1. `ModuleNotFoundError: No module named 'pypdf'`: dependency `pypdf` chưa được cài trong environment Python đang sử dụng để chạy pipeline.

2. `Report overwrite / 0.0 scores in ragas_report.json`: kết quả báo cáo RAGAS từng xuất hiện giá trị 0.0 do pipeline có cơ chế fallback khi xảy ra exception, đồng thời có logic xử lý file report ở thư mục root có thể khiến file cũ ghi đè lên artifact mới trong `reports/`.

### Nguyên nhân gốc rễ & Cách debug

**Debug Process:**

- Sử dụng `grep_search` và `view_file` để theo dõi chính xác luồng dữ liệu từ `evaluate_ragas()` → `save_report()` → `main.py`.
- Phát hiện `save_report()` vốn đã ghi trực tiếp vào `reports/ragas_report.json`.
- Logic `os.replace` trong `main.py` xử lý file ở root có thể di chuyển file cũ và ghi đè lên kết quả mới được sinh ra trong `reports/`.
- Sửa logic trong `main.py` để ưu tiên bảo tồn file báo cáo mới trong `reports/` và chỉ xử lý file root khi cần thiết.
- Qua quá trình debug, rút ra rằng artifact path và lifecycle của report cần được quản lý rõ ràng trong production pipeline.

### Kiến thức bổ sung & Bài học kinh nghiệm

- Cần quản lý chặt chẽ dependency và Python environment khi chạy pipeline production.
- Phải tách biệt rõ ràng giữa logic ghi báo cáo thực tế và logic fallback/exception.
- Artifact output cần có một source of truth duy nhất để tránh hiện tượng ghi đè hoặc sử dụng nhầm kết quả cũ.
- Khi debug pipeline end-to-end, không nên chỉ nhìn vào unit test của từng module mà cần kiểm tra cả integration flow và artifact cuối cùng.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Hỏi đáp Quy trình & Văn bản Nội bộ Doanh nghiệp (Enterprise Policy RAG)

### 1. Hiện trạng

- **Pipeline hiện tại:** Naive RAG sử dụng Naive Paragraph Chunking + Vector Search (Cosine Similarity) + Prompt LLM cơ bản.
- **Vấn đề / Bottlenecks đang gặp:**
  - Retrieval Precision thấp đối với các văn bản dài (quy trình mua sắm, quy chế tài chính).
  - Xuất hiện hallucination khi câu hỏi liên quan đến nhiều điều khoản rải rác.
  - Các bảng biểu (bảng ngạch lương, hạn mức phê duyệt) bị xé nhỏ làm mất ngữ cảnh hàng/cột.

### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Áp dụng **Hierarchical Parent-Child Chunking** cho các văn bản chính sách dạng văn bản, kết hợp **Structure-aware Chunking** giữ nguyên cấu trúc bảng biểu cho tài liệu định mức/bảng lương.

2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense Search với BGE-M3)** kết hợp **Reciprocal Rank Fusion (RRF)** để vừa khớp từ khóa thuật ngữ doanh nghiệp vừa hiểu ngữ nghĩa.

3. **Reranking:** Sử dụng **Cross-Encoder Reranker (`bge-reranker-large`)** để lọc Top-20 candidates xuống Top-5 context cô đọng nhất trước khi đưa vào LLM.

4. **Evaluation:** Tích hợp **RAGAS CI/CD pipeline** chạy tự động trên tập test 50+ câu hỏi benchmark hàng tuần để theo dõi Faithfulness ≥ 0.85 và Context Precision ≥ 0.90.

5. **Enrichment:** Sử dụng **Contextual Prepend (M5)** gắn thêm tên phòng ban và tên quy chế vào đầu mỗi chunk trước khi đánh index.

### 3. Timeline triển khai

- **Tuần 1:** Thiết kế lại Data Ingestion Pipeline (M1 Hierarchical Chunking + M5 Context Enrichment) và re-index toàn bộ kho tài liệu.
- **Tuần 2:** Triển khai M2 Hybrid Search (BM25 + Dense + RRF) & M3 Reranker; tích hợp bộ đo lường M4 RAGAS tự động.
- **Tuần 3:** Tối ưu hóa LLM System Prompt, thực hiện Failure Analysis trên các case study khó và đóng gói ứng dụng đưa vào Staging environment.
