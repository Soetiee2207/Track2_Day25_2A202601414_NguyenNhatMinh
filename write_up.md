# NimbusAI GPU FinOps — Báo cáo Phân tích Chi phí
### Lab 25 · AICB Phase 2 · Track 2 · Day 25
**Tác giả:** Nguyễn Nhật Minh — 2A202601414  
**Ngày:** 2026-08-28  
**Dữ liệu:** Telemetry tổng hợp tháng 6/2026 (seed=25)

---

## 1. Baseline vs. Optimized — Tổng quan chi phí

### Số liệu tổng hợp

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Chi phí hàng ngày (Inference) | $48.87/ngày | $8.48/ngày | **-82.6%** |
| Chi phí hàng tháng (Tổng) | $27,133/tháng | $14,626/tháng | **-46.1%** |
| Hiệu quả ($/1M-token) | $6.488/1M-token | $1.126/1M-token | **-82.7%** |
| Tiết kiệm tuyệt đối | — | — | **$12,507/tháng** |

### Phân rã tiết kiệm theo đòn bẩy

| Đòn bẩy | Tiết kiệm (USD/tháng) | % đóng góp |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | **80.3%** |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| **Tổng** | **$12,507** | **100%** |

> **Nhận xét:** Đòn bẩy Purchasing đóng góp tới 80% tổng tiết kiệm — đây là "quick win" lớn nhất cho NimbusAI vì không đòi hỏi thay đổi code, chỉ cần chuyển từ on-demand sang spot/reserved đúng chỗ.

---

## 2. Phân tích từng đòn bẩy chi phí

### 2.1 Purchasing Strategy (spot/reserved) — Tiết kiệm $10,040/tháng (80.3%)

**Trước tối ưu:** Tất cả 8 job chạy on-demand = $25,667/tháng  
**Sau tối ưu:** Spot + Reserved = $15,627/tháng → **tiết kiệm 39.1%**

Chi tiết từng job:

| Job | GPU | Tier | On-demand | Optimized | Saved |
|---|---|---|---|---|---|
| job-train-llm | H100 | spot | $12,000 | $7,596 | 36.7% |
| job-train-embed | A100 | spot | $2,148 | $1,393 | 35.1% |
| job-finetune | H100 | spot | $900 | $570 | 36.7% |
| job-infer-chat | A10G | reserved | $4,320 | $2,592 | **40.0%** |
| job-infer-rag | A100 | reserved | $3,866 | $2,160 | **44.1%** |
| job-infer-search | L4 | reserved | $1,728 | $972 | **43.8%** |
| job-dev-sandbox | A10G | spot | $480 | $203 | 57.7% |
| job-batch-eval | H100 | spot | $225 | $142 | 36.9% |

**Nguyên tắc lựa chọn tier:**
- Training jobs (`interruptible=True`) → **Spot**: chấp nhận gián đoạn + checkpoint, tiết kiệm ~37–58%
- Inference jobs chạy 24/7 (`duty cycle > 55%`) → **Reserved**: cam kết dài hạn, tiết kiệm ~40–44%
- Break-even: Reserved có lợi khi utilization ≥ 55% (13.2 giờ/ngày) với discount 45%

**Lý do spot training được chọn thay vì reserved:**  
Training LLM có checkpoint định kỳ → khi spot bị thu hồi, chỉ mất tối đa 1 checkpoint (vài giờ). Chi phí restart nhỏ hơn nhiều so với premium on-demand.

### 2.2 Inference Cost Levers — Tiết kiệm 82.6% chi phí inference

**Kết quả M2:**
```
Baseline:   $48.87/ngày   $6.488/1M-token
Optimized:  $ 8.48/ngày   $1.126/1M-token
Savings:    82.6%
```

Ba đòn bẩy được áp dụng tuần tự:

**a) Cascade (Model routing):**
- Request đơn giản → model nhỏ (A10G/L4) thay vì H100
- Giá model nhỏ rẻ hơn ~15× về $/1M-token
- Cascade là đòn bẩy mạnh nhất về tiết kiệm tuyệt đối

**b) Prompt Caching:**
- Input token đã cache → chỉ tính 10% giá (giảm 90%)
- Đặc biệt hiệu quả với chatbot có system prompt dài hoặc RAG context
- `discount_stack(batch=False, cache_hit_frac=1.0) = 0.10` → tiết kiệm 90%

**c) Batch API:**
- Request không cần real-time (eval, embedding, nightly jobs) → gộp batch
- Giảm 50% chi phí cố định
- **Khi KHÔNG dùng batch:** Chat assistant, search real-time, API latency-sensitive

**Tổng discount stack tối đa:**  
`batch=True + cache 100% → 0.50 × 0.10 = 0.05` → chỉ trả 5% giá gốc!

### 2.3 Right-size Util-lies — Tiết kiệm $655/tháng

**Hành động:** Hạ cấp GPU bị "lie" xuống tier thấp hơn phù hợp với workload thực tế.

`gpu-h100-4` thực sự chỉ cần MFU ~19.4% → có thể thay bằng A100 hoặc A10G cho inference decode task mà không mất throughput thực tế.

### 2.4 Kill Idle GPUs — Tiết kiệm $600/tháng

`gpu-h100-5` có 8 giờ idle/ngày = $20/ngày × 30 ngày = $600/tháng bỏ phí.  
Hành động: Implement auto-shutdown khi GPU utilization < 10% trong 30 phút liên tục.

---

## 3. Phân tích GPU-Util Lie

### GPU bị phát hiện

| GPU | Type | GPU-Util | MFU thực | Chênh lệch | Nguyên nhân nghi ngờ |
|---|---|---|---|---|---|
| `gpu-h100-4` | H100 | **98.2%** | **0.194 (19.4%)** | 5× | Memory stall / small batch |
| `gpu-a10g-1` | A10G | **96.9%** | **0.268 (26.8%)** | 3.6× | Kernel overhead / I/O bottleneck |

### Giải thích cơ chế

```
nvidia-smi GPU-Util = "GPU SM clock đang chạy không?" → TRUE khi có bất kỳ kernel nào active
MFU              = achieved_TFLOPs / peak_TFLOPs       → đo hiệu quả tính toán thực sự

Vì sao nvidia-smi "nói dối"?
  1. Memory stall: GPU clock đang chạy nhưng SM ngồi chờ dữ liệu từ HBM
     → LLM decode chỉ ~1–2 FLOP/byte (memory-bound), H100 ridge_point ≈ 295 FLOP/byte
     → GPU chỉ tận dụng được <10% compute, phần còn lại là "stall time"
  
  2. Kernel launch overhead: Nhiều kernel nhỏ phát liên tục
     → Mỗi lần launch kernel mất vài microseconds overhead
     → GPU "bận" schedule kernel nhưng không tính toán
  
  3. Small batch inference: Batch size nhỏ → nhiều SM không có việc làm
     → nvidia-smi thấy 1 SM bận → báo GPU "active"
```

### Tác động tài chính

```
gpu-h100-4: H100 on-demand ~$2.50/giờ
  Bạn trả: 1.0 × $2.50 = $2.50/giờ   (100% giá H100)
  Bạn nhận: 0.194 × peak FLOPs        (~20% FLOPs thực)
  Lãng phí: 80% chi phí H100 = $2.00/giờ = $1,440/tháng CHỈ TỪ GPU NÀY

gpu-a10g-1: A10G on-demand ~$0.75/giờ
  Bạn trả: 100% giá A10G
  Bạn nhận: ~27% FLOPs thực
  Lãng phí: 73% chi phí = $0.55/giờ = $396/tháng
```

**Kết luận:** `nvidia-smi GPU-Util` là chỉ số nguy hiểm nếu dùng để đánh giá hiệu quả. Luôn đo MFU (cho compute-bound) hoặc MBU (cho memory-bound) song song.

---

## 4. Phân bổ chi phí (Mission 4)

### Chi phí theo team ($/ngày)

| Team | Chi phí/ngày | % |
|---|---|---|
| assistant | $2.59 | 30.7% |
| search | $2.49 | 29.5% |
| eval | $1.79 | 21.2% |
| rag | $1.60 | 19.0% |
| **Tổng** | **$8.47** | **100%** |

**Tag coverage: 92%** → Vượt ngưỡng 80% → ✅ **Chargeback ready!**

8% request chưa có tag project (có team nhưng thiếu project tag). Nên fix để đạt 100% coverage trước khi implement chargeback thật.

### FOCUS Export

50 rows được xuất ra `focus_export.csv` theo chuẩn FinOps Foundation FOCUS.  
Ưu điểm: File này có thể import thẳng vào AWS Cost Explorer, Azure Cost Management, hoặc bất kỳ FinOps tool nào hỗ trợ FOCUS.

---

## 5. Sustainability — Carbon & Năng lượng

| Chỉ số | Giá trị |
|---|---|
| Năng lượng mỗi query | **0.24 Wh** |
| Carbon mỗi query | **0.091 gCO2e** |
| Vùng tốt nhất | **europe-north1** (Na Uy) |

### So sánh carbon theo vùng

| Vùng | Carbon intensity | Đánh giá |
|---|---|---|
| europe-north1 (Na Uy) | ~20 gCO2/kWh | ✅ Tốt nhất (thủy điện) |
| us-west-2 (Oregon) | ~150 gCO2/kWh | Tốt |
| us-east-1 (Virginia) | ~400 gCO2/kWh | Trung bình |
| asia-east1 (Đài Loan) | ~520 gCO2/kWh | Kém |
| europe-central2 (Ba Lan) | ~660 gCO2/kWh | ❌ Tệ nhất (than đá) |

**Khuyến nghị:** Di chuyển training jobs `interruptible=True` sang europe-north1 có thể giảm carbon **>95%** so với europe-central2. Với latency-sensitive inference, cần cân bằng giữa độ sạch và độ trễ mạng.

---

## 6. Khuyến nghị cho NimbusAI

*Với vai trò FinOps Lead, 3 hành động ưu tiên đầu tiên:*

### Hành động 1 (Ưu tiên cao nhất) — Chuyển sang Spot/Reserved ngay

**ROI:** $10,040/tháng tiết kiệm, không cần thay đổi code gì cả.

```
Tuần 1: Chuyển job-infer-chat, job-infer-rag, job-infer-search sang Reserved 1yr
         → Tiết kiệm $5,724/tháng ngay lập tức

Tuần 2: Chuyển training jobs (train-llm, train-embed, finetune, batch-eval)
         sang Spot với checkpoint mỗi 30 phút
         → Tiết kiệm thêm $3,376/tháng
```

**Rủi ro:** Spot có thể bị thu hồi → Implement checkpoint tự động trước khi chuyển.

### Hành động 2 — Fix GPU-Util Lies với MFU Monitoring

**ROI:** $655/tháng + prevent tái phát ($1,800+ nếu scale up sai GPU)

```
1. Thêm alert: Nếu GPU-Util > 90% nhưng MFU < 30% → cảnh báo ngay
2. gpu-h100-4: Điều tra nguyên nhân (batch size? memory stall?)
              → Tăng batch size hoặc hạ cấp xuống A100/A10G
3. gpu-a10g-1: Tương tự, xem xét right-size hoặc optimize kernel
4. Thêm vào dashboard: MFU và MBU song song với GPU-Util
```

**Bài học:** Không bao giờ ra quyết định mua thêm GPU chỉ dựa trên GPU-Util.

### Hành động 3 — Implement Inference Cascade + Chargeback

**ROI:** $1,212/tháng inference + accountability culture trong team

```
Cascade:
  - Phân loại request theo độ phức tạp (simple/complex)
  - Route: simple → A10G/L4 (rẻ 15×), complex → H100
  - Bắt đầu với 30% traffic để test chất lượng

Chargeback:
  - Tag coverage đã đạt 92% → đủ điều kiện
  - Fix 8% request thiếu project tag (1 tuần)
  - Bắt đầu showback tháng 1, chuyển sang chargeback tháng 2
  - Khi team bị tính tiền thật → họ sẽ tự optimize!
```

---

## Tổng kết

| | Trước | Sau | Tiết kiệm |
|---|---|---|---|
| $/tháng | $27,133 | $14,626 | **$12,507 (46.1%)** |
| $/1M-token | $6.488 | $1.126 | **$5.362 (82.7%)** |
| GPU-Util Lie | 2 GPU | 0 GPU | Đã fix |
| Idle waste | $600/tháng | $0 | **Đã fix** |
| Tag coverage | 92% | 92% | Chargeback ready |

**Kết luận:** NimbusAI có thể tiết kiệm **$12,500+/tháng (~$150,000/năm)** chỉ với 3 hành động không cần viết thêm model code, không cần mua GPU mới. FinOps không phải cắt giảm chất lượng — mà là **trả đúng tiền cho đúng giá trị nhận được**.

---

*Dữ liệu: NimbusAI synthetic telemetry, June 2026 snapshot*  
*Tool: Lab 25 GPU FinOps Framework (pandas + matplotlib)*  
*Tác giả: Nguyễn Nhật Minh — 2A202601414 — AICB Phase 2 Track 2*
