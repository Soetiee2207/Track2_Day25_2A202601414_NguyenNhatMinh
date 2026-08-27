# 📘 Guild Lab 25 — GPU FinOps Optimization
## Hướng dẫn Hoàn thành từ A đến Z

> **Dành cho:** Sinh viên AICB · Phase 2 · Track 2 · Day 25
> **Tác giả:** Nguyễn Nhật Minh — 2A202601414
> **Thời gian ước tính:** 3–5 giờ (bao gồm 2 extensions)
> **Yêu cầu:** Python 3.9+, không cần GPU, không cần tài khoản cloud, không cần API key

---

## 📋 Mục lục

1. [Yêu cầu hệ thống & Cài đặt ban đầu](#1-yêu-cầu-hệ-thống)
2. [Hiểu bối cảnh bài lab trước khi code](#2-bối-cảnh)
3. [Cài đặt môi trường Python](#3-môi-trường)
4. [Khám phá cấu trúc dự án & dữ liệu](#4-dữ-liệu)
5. [Mission 1 — Kiểm toán hiệu quả GPU](#5-mission-1)
6. [Mission 2 — Đòn bẩy chi phí Inference](#6-mission-2)
7. [Mission 3 — Chiến lược mua GPU](#7-mission-3)
8. [Mission 4 — Phân bổ chi phí](#8-mission-4)
9. [Mission 5 — Báo cáo tổng hợp](#9-mission-5)
10. [Kiểm tra toàn diện (verify + pytest)](#10-kiểm-tra)
11. [Phần mở rộng Your Turn (bắt buộc ≥2)](#11-extension)
12. [Chuẩn bị nộp bài](#12-nộp-bài)
13. [Bảng tra lỗi thường gặp](#13-lỗi)
14. [Bảng điểm & Tiêu chí chấm](#14-bảng-điểm)

---

## 1. Yêu cầu hệ thống & Cài đặt ban đầu

### 1.1 Yêu cầu tối thiểu

| Yêu cầu | Chi tiết |
|---|---|
| **Hệ điều hành** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| **Python** | 3.9, 3.10, 3.11, hoặc 3.12 |
| **RAM** | ≥ 4 GB |
| **Disk** | ≥ 500 MB trống |
| **GPU** | ❌ KHÔNG cần |
| **Internet** | Cần khi cài thư viện lần đầu |
| **Cloud/API Key** | ❌ KHÔNG cần |

### 1.2 Kiểm tra Python đã cài chưa

```powershell
python --version   # Kỳ vọng: Python 3.9.x trở lên
pip --version      # Kỳ vọng: pip 23.x trở lên
```

> Nếu chưa có Python: Tải từ https://www.python.org/downloads/ và chọn Python 3.11.
> Khi cài trên Windows, nhớ tích vào ô **"Add Python to PATH"**.

### 1.3 Thư mục làm việc

Tất cả lệnh chạy từ thư mục gốc dự án:

```
E:\26.AI-VIN\Track 02\Track2_Day25_2A202601414_NguyenNhatMinh\
```

---

## 2. Hiểu bối cảnh bài lab trước khi code

> **ĐỌC PHẦN NÀY TRƯỚC** — Hiểu "tại sao" trước khi làm "thế nào"

### 2.1 Câu chuyện

Bạn là **FinOps Engineer** mới gia nhập **NimbusAI** — startup LLM đang có hóa đơn GPU mất kiểm soát. CEO giao cho bạn:

- Dữ liệu telemetry từ 11 GPU (24 giờ/GPU)
- Bảng giá 7 loại GPU tháng 6/2026
- Nhật ký 2,400 request LLM từ nhiều team

**Nhiệm vụ:** Tìm điểm lãng phí và **cắt giảm 40–95% chi phí**, đo bằng `$/1M-token`.

### 2.2 Tại sao `$/1M-token` quan trọng hơn `$/GPU-giờ`?

```
$/GPU-giờ  = Chi phí thuê GPU         → chỉ đo "bạn trả bao nhiêu"
$/1M-token = Chi phí mỗi triệu token  → đo "bạn nhận được gì"
```

**Ví dụ:** Hai team đều trả $2.50/giờ, nhưng Team A phục vụ 100K tokens/giờ còn Team B
phục vụ 1M tokens/giờ. `$/1M-token` của Team B thấp hơn **10 lần**.

### 2.3 Bảng từ điển thuật ngữ cốt lõi

| Thuật ngữ | Giải thích | Giá trị tốt |
|---|---|---|
| **MFU** | % FLOPs thực tế / FLOPs tối đa GPU — thước đo hiệu quả tính toán | 35–50% |
| **MBU** | % Băng thông thực tế / Băng thông tối đa — cho memory-bound | >50% |
| **GPU-Util** | Đo "clock có đang bật" từ nvidia-smi — KHÔNG đo hiệu quả! | ⚠️ Đừng tin |
| **Cascade** | Request đơn giản → model nhỏ rẻ hơn 15× | — |
| **Prompt Caching** | Phần input đã cache → giảm 90% chi phí phần đó | — |
| **Batch API** | Gộp request offline → giảm 50%, nhưng độ trễ cao hơn | — |
| **Spot Instance** | GPU rẻ ~60%, có thể bị thu hồi → cần checkpoint | — |
| **Reserved Instance** | Cam kết 1–3 năm → giảm ~45% | Cần ≥55% utilization |
| **Showback** | Hiển thị chi phí theo team để nhận thức (chưa thu tiền) | — |
| **Chargeback** | Thu tiền thật từ team theo mức dùng | Tag coverage ≥80% |
| **FOCUS** | Chuẩn mở đa cloud cho dữ liệu chi phí (FinOps Foundation) | — |

### 2.4 Sơ đồ tổng quan luồng tối ưu

```
PHÁT HIỆN LÃNG PHÍ (Mission 1)
  GPU-Util 98% ≠ MFU 98%  →  Phải đo MFU + MBU thực tế

          ↓

3 ĐÒN BẨY INFERENCE (Mission 2)
  1. Cascade: dễ → model nhỏ (rẻ 15×)
  2. Prompt Cache: cached input → 90% off
  3. Batch API: offline requests → 50% off

          ↓

CHIẾN LƯỢC MUA GPU (Mission 3)
  On-Demand / Spot (rẻ 60%) / Reserved (rẻ 45%)

          ↓

PHÂN BỔ CHI PHÍ (Mission 4)
  Visibility → Showback → Chargeback
  (cần tag coverage ≥80% để chargeback)

          ↓

BÁO CÁO TỔNG HỢP (Mission 5)
  Baseline vs. Optimized + Sustainability
```

---

## 3. Cài đặt môi trường Python

### Bước 3.1 — Vào thư mục dự án

```powershell
cd "E:\26.AI-VIN\Track 02\Track2_Day25_2A202601414_NguyenNhatMinh"
```

### Bước 3.2 — Tạo và kích hoạt Virtual Environment

```powershell
# Tạo venv
python -m venv .venv

# Kích hoạt — Windows PowerShell
.venv\Scripts\activate

# Kích hoạt — macOS/Linux
# source .venv/bin/activate
```

✅ **Dấu hiệu thành công:** Terminal hiển thị `(.venv)` ở đầu:
```
(.venv) PS E:\26.AI-VIN\Track 02\...>
```

### Bước 3.3 — Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

| Thư viện | Phiên bản | Dùng để |
|---|---|---|
| `pandas` | ≥ 2.0 | Đọc và xử lý file CSV |
| `matplotlib` | ≥ 3.7 | Vẽ biểu đồ savings waterfall |
| `pytest` | ≥ 7.4 | Chạy 15 unit + integration tests |

### Bước 3.4 — Sinh dữ liệu tổng hợp

```powershell
python data/generate.py
```

Lệnh tạo 4 file CSV với **seed cố định = 25** (luôn cho kết quả giống nhau).

**Output mong đợi:**
```
Generated data/gpu_telemetry.csv   (264 rows, 11 GPUs × 24 hours)
Generated data/token_usage.csv     (2400 rows)
Generated data/workloads.csv       (8 rows)
Generated data/price_catalog.csv   (7 rows)
```

### Bước 3.5 — Kiểm tra lần đầu

```powershell
python verify.py
```

> Lần đầu chưa pass 11/11 là **bình thường**. Hoàn thành từng Mission sẽ đạt dần.

---

## 4. Khám phá cấu trúc dự án & dữ liệu

### 4.1 Cấu trúc thư mục

```
Track2_Day25_2A202601414_NguyenNhatMinh/
├── data/                    ← Dữ liệu đầu vào (CSV)
│   ├── generate.py          ← Script sinh CSV (seed=25, tất định)
│   ├── price_catalog.csv    ← Giá 7 loại GPU
│   ├── gpu_telemetry.csv    ← 11 GPU × 24 giờ telemetry
│   ├── token_usage.csv      ← 2,400 request LLM
│   └── workloads.csv        ← 8 workload training/inference
├── finops/                  ← Engine tính toán (KHÔNG sửa trừ khi làm extension)
│   ├── metrics.py           ← MFU, MBU, roofline, flag util lies
│   ├── pricing.py           ← Chi phí request, $/1M-token, discount, tier
│   ├── allocation.py        ← Cost by tag, FOCUS export
│   ├── sustainability.py    ← Năng lượng, carbon, vùng tối ưu
│   └── report.py            ← Tạo báo cáo Markdown + biểu đồ
├── missions/                ← 5 bài tập chính
│   ├── m1_efficiency_audit.py
│   ├── m2_inference_levers.py
│   ├── m3_purchasing.py
│   ├── m4_allocation.py
│   ├── m5_report.py
│   └── run_all.py           ← Chạy M1→M5 liên tiếp
├── tests/                   ← 15 tests (KHÔNG ĐƯỢC SỬA!)
├── outputs/                 ← Kết quả sinh ra
├── verify.py                ← Kiểm tra 11 checks
└── requirements.txt
```

### 4.2 Khám phá từng file dữ liệu

```python
import pandas as pd

# --- Bảng giá GPU ---
df_price = pd.read_csv("data/price_catalog.csv")
print(df_price[["gpu_type", "on_demand_hr", "spot_hr",
                "reserved_3yr_hr", "peak_tflops_fp16", "watts"]])
# Chú ý: H100 on-demand ~$2.50/giờ; Spot rẻ hơn 40-60%; Reserved rẻ nhất

# --- Telemetry GPU (tìm GPU-Util Lie) ---
df_tel = pd.read_csv("data/gpu_telemetry.csv")
print(df_tel[["gpu_id", "gpu_type", "gpu_util_pct",
              "achieved_tflops", "achieved_bw_tbs"]].head(20))
# Tìm: gpu_util_pct cao (~98%) nhưng achieved_tflops thấp → GPU-Util Lie

# --- Token usage ---
df_tok = pd.read_csv("data/token_usage.csv")
print(df_tok[["route_tier", "input_tokens", "output_tokens",
              "cached_input_tokens", "is_batch", "team"]].head(10))
print("Tier:", df_tok["route_tier"].value_counts().to_dict())
print("Batch rate:", df_tok["is_batch"].mean())

# --- Workloads ---
df_work = pd.read_csv("data/workloads.csv")
print(df_work[["job_id", "gpu_type", "num_gpus", "hours_per_day", "interruptible"]])
# interruptible=1 → phù hợp spot; hours_per_day → quyết định reserved hay on-demand
```

---

## 5. Mission 1 — Kiểm toán hiệu quả GPU

**Mục tiêu:** Phát hiện GPU "nói dối" (GPU-Util cao nhưng MFU thấp) và tính lãng phí idle.

**Files:** `finops/metrics.py`, `missions/m1_efficiency_audit.py`

### 5.1 Đọc và hiểu `finops/metrics.py`

```python
# Tính MFU — thước đo hiệu quả tính toán thực sự
def compute_mfu(achieved_tflops, peak_tflops):
    """Trả về 0.0–1.0. Tốt: 0.35–0.50"""
    return achieved_tflops / peak_tflops

# Tính MBU — thước đo hiệu quả băng thông bộ nhớ (cho decode)
def compute_mbu(achieved_bw_tbs, peak_bw_tbs):
    return achieved_bw_tbs / peak_bw_tbs

# Phân loại roofline
def roofline_regime(arithmetic_intensity, ridge_point):
    """
    H100 ridge_point BF16 ≈ 295 FLOP/byte
    LLM decode: ~1–2 FLOP/byte  → memory-bound (MBU quan trọng hơn)
    LLM prefill: ~455 FLOP/byte → compute-bound (MFU quan trọng hơn)
    """
    return "memory-bound" if arithmetic_intensity < ridge_point else "compute-bound"

# Phát hiện GPU-Util Lie
def flag_util_lies(rows, util_threshold=0.90, mfu_threshold=0.30):
    """
    Điều kiện: (gpu_util_pct / 100) >= 0.90 VÀ MFU < 0.30
    ⚠️ QUAN TRỌNG: gpu_util_pct ở thang 0–100, phải chia 100 trước khi so sánh!
    """
    ...

# Chi phí lãng phí do GPU idle
def idle_waste_usd(idle_hours, on_demand_hr):
    return idle_hours * on_demand_hr
```

### 5.2 Tại sao "GPU-Util Lie" xảy ra?

```
nvidia-smi GPU-Util = "GPU clock có đang tick không?"
                    ≠ "GPU có đang tính toán hiệu quả không?"

Nguyên nhân GPU-Util cao nhưng MFU thấp:
  • Memory stall: GPU chờ dữ liệu từ HBM (băng thông tắc nghẽn)
  • Kernel launch overhead: Nhiều kernel nhỏ, overhead chiếm đa số
  • I/O bottleneck: GPU ngồi chờ đọc/ghi dữ liệu
  • Small batch: Batch nhỏ, nhiều SM không có việc làm

Kết quả thực tế:
  gpu-h100-4: GPU-Util = 98% NHƯNG MFU = 0.20 (~20%)
  → Bạn trả tiền cho 100% H100 nhưng chỉ nhận 1/5 FLOPs
```

### 5.3 Chạy Mission 1

```powershell
python missions/m1_efficiency_audit.py
```

**Output mong đợi:**
```
== M1 Efficiency Audit ==
GPU             type    util%    MFU    MBU  idle_h
gpu-a100-0      A100     72.0  0.385  0.621       0
gpu-a100-1      A100     68.0  0.361  0.582       0
gpu-h100-4      H100     98.0  0.202  0.450       0   ← GPU-Util LIE!
gpu-t4-8        T4        8.0  0.045  0.081      20   ← Idle waste
...

GPU-Util LIES (util>=90% but MFU<30%): ['gpu-h100-4']
Idle waste (1 day): $125.00  ->  ~$3,750/month
```

### 5.4 Phân tích câu hỏi

1. **GPU nào bị "lie"?** → `gpu-h100-4`: 98% Util nhưng MFU chỉ 20%
2. **Tác động tài chính:** Trả tiền H100 full giờ nhưng chỉ nhận 1/5 hiệu năng
3. **Lãng phí idle:** ~$125/ngày → ~$3,750/tháng → ~$45,000/năm

---

## 6. Mission 2 — Đòn bẩy chi phí Inference

**Mục tiêu:** Áp dụng 3 đòn bẩy (Cascade, Cache, Batch) để giảm `$/1M-token`.

**Files:** `finops/pricing.py`, `missions/m2_inference_levers.py`

### 6.1 Các hàm cốt lõi trong `finops/pricing.py`

```python
def request_cost(input_tok, output_tok, price_in_per_m, price_out_per_m,
                 cached_in=0, batch=False):
    """
    Công thức chi tiết:
      non_cached_in = input_tok - cached_in
      cost_in = (non_cached_in / 1e6 * price_in_per_m)
              + (cached_in / 1e6 * price_in_per_m * 0.10)   ← 90% off!
      cost_out = output_tok / 1e6 * price_out_per_m
      total = (cost_in + cost_out) * (0.50 if batch else 1.0)  ← 50% off!
    """

def dollars_per_million(total_cost_usd, total_tokens):
    """$/1M-token = total_cost / total_tokens * 1,000,000"""

def discount_stack(batch=False, cache_hit_frac=0.0):
    """
    Chiết khấu nhân lên nhau:
    factor = (0.50 if batch else 1.0) * (1.0 - 0.90 * cache_hit_frac)
    
    Ví dụ: batch=True + cache 100%
      = 0.50 * (1.0 - 0.90 * 1.0) = 0.50 * 0.10 = 0.05
      → Chỉ trả 5% giá gốc! Tiết kiệm 95%!
    """
```

### 6.2 Tính tay để hiểu (test trong Python)

```python
from finops.pricing import request_cost, discount_stack, dollars_per_million

# BASELINE: model lớn, không cache, không batch
baseline = request_cost(
    input_tok=1000, output_tok=200,
    price_in_per_m=3.00,    # Model lớn: $3/1M input
    price_out_per_m=15.00   # Model lớn: $15/1M output
)
print(f"Baseline: ${baseline:.5f}")
# = (1000/1e6 * 3.00) + (200/1e6 * 15.00) = $0.003 + $0.003 = $0.006

# OPTIMIZED: model nhỏ + 80% cache + batch
optimized = request_cost(
    input_tok=1000, output_tok=200,
    price_in_per_m=0.20,    # Model nhỏ: $0.20/1M (rẻ 15×!)
    price_out_per_m=0.40,   # Model nhỏ: $0.40/1M
    cached_in=800,          # 80% input đã cache → giảm 90% phần này
    batch=True              # Batch → giảm 50%
)
print(f"Optimized: ${optimized:.8f}")  # Rất nhỏ

# DISCOUNT STACK
print(f"Batch + 80% cache:  {discount_stack(True, 0.8):.3f}")   # ~0.14 = 14%
print(f"Batch + 100% cache: {discount_stack(True, 1.0):.3f}")   # 0.050 = 5%
```

### 6.3 Chạy Mission 2

```powershell
python missions/m2_inference_levers.py
```

**Output mong đợi:**
```
== M2 Inference Cost Levers ==
requests=2400  tokens=1,234,567
baseline      : $45.23/day   $36.643/1M-token
after cascade : $28.10/day  (37.8% saved)  ← dùng model nhỏ hơn
after cache   : $18.50/day  (59.1% saved)  ← cache prefix input
after batch   : $12.81/day  (71.7% saved)  ← gộp offline requests
optimized     : $12.81/day   $10.374/1M-token
savings       : 71.7%  (cascade + caching + batch)
discount stack (batch + 100% cache): 0.050 of naive
```

### 6.4 Câu hỏi phân tích

1. **Đòn bẩy mạnh nhất:** Cascade — giảm price/token đến 15× (từ $3→$0.20/1M)
2. **Khi nào KHÔNG dùng batch:** Chatbot real-time, autocomplete, live support
3. **`discount_stack(True, 1.0) = 0.05`:** Batch 50% × Cache 100% (10%) = 5%

---

## 7. Mission 3 — Chiến lược mua GPU

**Mục tiêu:** Chọn đúng tier (on-demand / spot / reserved) cho từng workload.

**Files:** `finops/pricing.py`, `missions/m3_purchasing.py`

### 7.1 Logic `recommend_tier()`

```python
def break_even_utilization(reserved_discount):
    """
    Reserved có lợi khi: utilization >= 1 - discount
    Với discount=45%: break_even = 55% = 13.2 giờ/ngày
    """
    return 1.0 - reserved_discount  # = 0.55

def recommend_tier(hours_per_day, interruptible, reserved_discount=0.45):
    duty = hours_per_day / 24.0
    be = break_even_utilization(reserved_discount)   # = 0.55

    if interruptible and hours_per_day < 24:
        return "spot"       # Có thể gián đoạn + checkpoint → spot
    if duty >= be:
        return "reserved"   # ≥13.2 giờ/ngày → reserved có lợi
    return "on_demand"      # Còn lại → on-demand linh hoạt
```

**Bảng quyết định tier:**

| Điều kiện | Tier | Lý do |
|---|---|---|
| `interruptible=1` + < 24h/ngày | **Spot** | Rẻ ~60%, chấp nhận gián đoạn |
| Duty ≥ 55% (≥13.2h/ngày) | **Reserved** | Cam kết dài hạn, tiết kiệm 45% |
| Duty < 55%, không gián đoạn | **On-Demand** | Linh hoạt, không cam kết |

### 7.2 Spot Checkpoint Simulation

```python
from finops.pricing import spot_checkpoint_cost

result = spot_checkpoint_cost(
    job_hours=720,           # 30 ngày × 24h
    spot_hr=1.40,            # H100 spot price
    on_demand_hr=2.50,       # H100 on-demand price
    interrupt_rate=0.05,     # 5% xác suất thu hồi mỗi giờ
    ckpt_overhead_frac=0.03  # 3% overhead ghi checkpoint
)
print(result)
# {'spot_effective_hours': 759.6,   ← nhiều hơn vì restart sau thu hồi
#  'spot_cost': 1063.44,
#  'on_demand_cost': 1800.0,
#  'savings_pct': 40.9}              ← tiết kiệm ~41%

# "effective_hours > job_hours": bị thu hồi → restart từ checkpoint
# → tổng runtime dài hơn, nhưng giá spot rẻ hơn đủ để vẫn tiết kiệm
```

### 7.3 Chạy Mission 3

```powershell
python missions/m3_purchasing.py
```

**Output mong đợi:**
```
== M3 Purchasing Strategy ==
break-even utilization @ 45% reserved discount = 55%

job              gpu    h/day  int  tier        on-demand    optimized  saved
training-001     H100    20     1   spot         $15,000      $7,200    52.0%
training-002     A100    24     0   reserved     $12,000      $6,600    45.0%
inference-001    H100    16     0   reserved     $10,800      $5,940    45.0%
inference-002    T4       8     1   spot          $3,600      $1,728    52.0%
...

monthly: on-demand $45,000  ->  optimized $25,200  (44.0% saved)
```

---

## 8. Mission 4 — Phân bổ chi phí

**Mục tiêu:** Chuyển hóa đơn GPU tổng thành trách nhiệm theo team, xuất chuẩn FOCUS.

**Files:** `finops/allocation.py`, `missions/m4_allocation.py`

### 8.1 Lý thuyết phân bổ chi phí

```
Thang trưởng thành FinOps:

1. Visibility (Nhìn thấy)
   → Biết tổng chi phí GPU

2. Showback (Thông báo)
   → Biết team nào dùng bao nhiêu, thông báo để họ biết
   → Chưa thu tiền

3. Chargeback (Thu tiền thật)
   → Thực sự thu tiền từ team theo mức dùng
   → YÊU CẦU: Tag coverage >= 80%
   → Lý do: Nếu 30% request không có tag → tính phí không công bằng
```

### 8.2 Các hàm trong `finops/allocation.py`

```python
def cost_by_tag(records, tag_key, price_in_per_m, price_out_per_m):
    """Tổng hợp chi phí theo team hoặc project"""

def tag_coverage(records, tag_key):
    """% records có tag hợp lệ (không None/empty)"""

def chargeback_ready(coverage, threshold=0.80):
    """True nếu coverage >= 80%"""
    return coverage >= threshold

def to_focus_rows(records, billing_account_id, charge_period):
    """
    Xuất sang chuẩn FOCUS (FinOps Foundation)
    Cột bắt buộc: BillingAccountId, ChargePeriodStart,
                  ServiceCategory, BilledCost, team, project
    """
```

### 8.3 Chạy Mission 4

```powershell
python missions/m4_allocation.py
```

**Output mong đợi:**
```
== M4 Cost Allocation ==
cost by team ($/day):
  ml-research    $    8.45   (64.2%)
  platform       $    3.12   (23.7%)
  product        $    1.24    (9.4%)
  [untagged]     $    0.35    (2.7%)

tag coverage: 92%  ->  chargeback ready? True

FOCUS export -> outputs/focus_export.csv (50 rows)
```

### 8.4 Kiểm tra FOCUS export

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/focus_export.csv'); print(df.head())"
```

**Cột FOCUS quan trọng:**

| Cột | Ý nghĩa |
|---|---|
| `BillingAccountId` | ID tài khoản thanh toán |
| `ChargePeriodStart` | Ngày bắt đầu kỳ tính phí |
| `ServiceCategory` | `AI and Machine Learning` |
| `BilledCost` | Chi phí thực tế |
| `team`, `project` | Tags phân bổ |

**FOCUS quan trọng vì:** Chuẩn mở — hoạt động với AWS, Azure, GCP, on-prem.

---

## 9. Mission 5 — Báo cáo tổng hợp

**Mục tiêu:** Gộp M1–M4, tạo báo cáo baseline vs. optimized với biểu đồ + sustainability.

**Files:** `finops/report.py`, `finops/sustainability.py`, `missions/m5_report.py`

### 9.1 Bốn đòn bẩy trong M5

```
Baseline spend
  ↓ 1. Inference (cascade + cache + batch)   ← M2 (~70% savings inference)
  ↓ 2. Purchasing (spot + reserved)          ← M3 (~44% savings GPU cost)
  ↓ 3. Right-size util-lies                  ← M1 (hạ cấp GPU bị "lie")
  ↓ 4. Kill idle GPUs                        ← M1 (tắt GPU util < 10%)
Optimized spend
```

### 9.2 Sustainability trong `finops/sustainability.py`

```python
from finops.sustainability import wh_per_query, carbon_g, REGION_CARBON

# Tính năng lượng tiêu thụ mỗi query
wh = wh_per_query(tokens=200, power_w=700, tok_per_sec=50)
print(f"Energy: {wh:.4f} Wh per query")

# Carbon theo vùng triển khai
# europe-north1 (Na Uy): ~20 gCO2/kWh   ← sạch nhất (thủy điện)
# us-east-1 (Virginia):  ~400 gCO2/kWh
# europe-central2 (Ba Lan): ~660 gCO2/kWh  ← dơ nhất (than đá)
for region, intensity in REGION_CARBON.items():
    co2 = carbon_g(wh, intensity)
    print(f"  {region}: {co2:.3f} gCO2e")
```

### 9.3 Chạy Mission 5

```powershell
python missions/m5_report.py
```

**Output mong đợi:**
```
== M5 Optimization Report ==

NimbusAI — GPU Cost Optimization Report
Baseline spend:    $27,133/month
Optimized spend:   $14,626/month
Projected savings: $12,507 (46%)

Savings breakdown:
  Inference levers:    $8,123  (29.9%)
  Purchasing strategy: $3,960  (14.6%)
  Right-size lies:       $424   (1.6%)

Sustainability:
  Energy per query: 0.24 Wh
  Carbon per query: 0.091 gCO2e
  Best region: europe-north1

Written: outputs/report.md + outputs/savings.png
```

### 9.4 Kiểm tra báo cáo output

```powershell
type outputs\report.md    # Đọc nội dung
start outputs\savings.png  # Mở biểu đồ (Windows)
```

**Báo cáo phải có đủ:**
- [ ] Baseline spend, Optimized spend, % tiết kiệm tổng
- [ ] Breakdown từng lever với số tiền cụ thể
- [ ] Sustainability: năng lượng/query, carbon/query, vùng tốt nhất

---

## 10. Kiểm tra toàn diện (verify + pytest)

### 10.1 Chạy tất cả missions cùng lúc

```powershell
python missions/run_all.py
```

### 10.2 Chạy verify.py — mục tiêu 11/11

```powershell
python verify.py
```

**Output mong đợi (đạt điểm tối đa):**
```
============================================================
  LAB 25 VERIFY
============================================================
  [PASS] M1 flags the GPU-Util lie (gpu-h100-4)
  [PASS] M1 detects idle waste
  [PASS] M2 $/1M-token drops after optimization
  [PASS] M2 inference savings in 60-95% band
  [PASS] M3 recommends a spot tier
  [PASS] M3 recommends a reserved tier
  [PASS] M3 purchasing saves money
  [PASS] M4 tag coverage 85-100%
  [PASS] M4 chargeback gate is open
  [PASS] M5 total savings in 40-95% band
  [PASS] M5 report.md written
------------------------------------------------------------
  11/11 checks passed
============================================================
```

**Nếu fail check:**

| Check fail | Nguyên nhân thường gặp | Cách sửa |
|---|---|---|
| M1 flags lie | So sánh sai đơn vị | `util = gpu_util_pct / 100.0` rồi so với 0.90 |
| M2 savings out of band | Discount áp dụng sai | `cache_discount=0.10`, `batch_discount=0.50` |
| M3 no spot/reserved | Thiếu nhánh logic tier | Thêm cả hai nhánh spot và reserved |
| M4 coverage < 85% | Data chưa generate đúng | Xóa CSV cũ + `python data/generate.py` |
| M5 savings out of band | Kết quả M1–M4 sai | Kiểm tra từng mission phía trước |
| M5 report not written | Thư mục outputs thiếu | `mkdir outputs` |

### 10.3 Chạy pytest — mục tiêu 15/15

```powershell
pytest -q
```

**Output mong đợi:**
```
...............
15 passed in 0.42s
```

```powershell
# Nếu có lỗi — verbose mode
pytest -v

# Chạy chỉ một file test
pytest tests/test_metrics.py -v

# Chạy chỉ một test cụ thể
pytest tests/test_pricing.py::test_discount_stack -v
```

**Danh sách test theo file:**

| File | Tests | Kiểm tra |
|---|---|---|
| `test_metrics.py` | 4 tests | MFU/MBU đúng, flag_util_lies đúng ngưỡng, idle_waste đúng |
| `test_pricing.py` | 5 tests | request_cost, discount_stack, break_even, recommend_tier |
| `test_allocation.py` | 4 tests | cost_by_tag, tag_coverage, chargeback_ready, FOCUS format |
| `test_report.py` | 1 test | build_report tạo Markdown đúng section |
| `test_data_and_missions.py` | 1 test | End-to-end M1→M5 pipeline |

> **CẢNH BÁO:** KHÔNG được sửa file test! Bị phát hiện → mất toàn bộ 20 điểm phần B.

---

## 11. Phần mở rộng Your Turn (bắt buộc ≥2)

Cần làm **ít nhất 2** trong 5 extensions để đạt 20 điểm phần D. Mỗi extension tối đa 10 điểm.

---

### Extension 3 — `cache_is_worth_it()` ⭐ KHUYẾN NGHỊ

**Ý tưởng:** Cache không phải lúc nào cũng có lợi. Cần đủ số lần đọc lại để bù chi phí ghi.

**File sửa:** `finops/pricing.py`

**Bước 1 — Thêm hàm:**

```python
def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10
) -> bool:
    """
    Cache tiết kiệm tiền khi: tổng savings từ đọc > chi phí ghi
    
    Mỗi lần đọc (thay vì tính lại), tiết kiệm: price_in * (1 - read_discount) per token
    Break-even: N_reads >= 1 / (1 - read_discount)
    
    Với read_discount=0.10 (90% off khi đọc cache):
      break_even = 1 / (1 - 0.10) = 1.11 reads
      → Chỉ cần đọc lại hơn 1 lần là đã có lợi!
    """
    if write_cost_per_m <= 0:
        return True  # Ghi miễn phí → luôn có lợi
    breakeven_reads = 1.0 / (1.0 - read_discount)
    return avg_cache_reads >= breakeven_reads
```

**Bước 2 — Áp dụng trong `missions/m2_inference_levers.py`:**

```python
import pandas as pd
from finops.pricing import cache_is_worth_it

df = pd.read_csv("data/token_usage.csv")
cached_df = df[df["cached_input_tokens"] > 0]

# Ước tính avg_cache_reads từ dữ liệu
# (mỗi cached prefix được dùng trung bình bao nhiêu lần)
total_cached_requests = len(cached_df)
unique_cache_entries = df["cached_input_tokens"].nunique()
avg_reads = total_cached_requests / max(1, unique_cache_entries)

write_cost_per_m = 0.375  # Chi phí lưu cache (ví dụ Gemini)

print("\n=== Extension 3: Cache Economics ===")
breakeven = 1.0 / (1.0 - 0.10)
print(f"Break-even reads needed: {breakeven:.2f}")
print(f"Actual avg cache reads:  {avg_reads:.2f}")
worth = cache_is_worth_it(avg_reads, write_cost_per_m)
print(f"Cache is worth it? {worth}")

if worth:
    print("-> Cache savings INCLUDED in M2 optimization")
else:
    print("-> Cache savings EXCLUDED (not economical)")
```

**Đo lường:** So sánh M2 savings với và không có điều kiện `cache_is_worth_it()`.

---

### Extension 4 — Ngân sách Reasoning ⭐ KHUYẾN NGHỊ

**Ý tưởng:** `is_reasoning=1` tốn năng lượng ~80× và chi phí cao hơn nhiều. Cần routing rule.

**File sửa:** Thêm vào cuối `missions/m2_inference_levers.py`

```python
import pandas as pd
from finops.pricing import request_cost, dollars_per_million
from finops.sustainability import wh_per_query

df = pd.read_csv("data/token_usage.csv")
reasoning_df = df[df["is_reasoning"] == 1]
normal_df    = df[df["is_reasoning"] == 0]

print("\n=== Extension 4: Reasoning Budget Analysis ===")
print(f"Total requests:     {len(df)}")
print(f"Reasoning requests: {len(reasoning_df)} ({len(reasoning_df)/len(df):.1%})")
print(f"Normal requests:    {len(normal_df)} ({len(normal_df)/len(df):.1%})")

# Giá reasoning cao hơn ~3-5× (Extended Thinking model)
PRICE_IN_REASONING  = 8.00
PRICE_OUT_REASONING = 24.00
PRICE_IN_NORMAL     = 3.00
PRICE_OUT_NORMAL    = 15.00

def calc_cost(group_df, price_in, price_out):
    total_cost, total_tokens = 0, 0
    for _, row in group_df.iterrows():
        cost = request_cost(
            input_tok=int(row["input_tokens"]),
            output_tok=int(row["output_tokens"]),
            price_in_per_m=price_in,
            price_out_per_m=price_out,
            cached_in=int(row.get("cached_input_tokens", 0)),
            batch=bool(int(row.get("is_batch", 0)))
        )
        total_cost += cost
        total_tokens += int(row["input_tokens"]) + int(row["output_tokens"])
    return total_cost, total_tokens

cost_r, tok_r = calc_cost(reasoning_df, PRICE_IN_REASONING, PRICE_OUT_REASONING)
cost_n, tok_n = calc_cost(normal_df, PRICE_IN_NORMAL, PRICE_OUT_NORMAL)
total_cost = cost_r + cost_n

print(f"\nReasoning cost: ${cost_r:.2f}/day ({cost_r/total_cost:.1%} of total)")
print(f"Normal cost:    ${cost_n:.2f}/day")
print(f"Reasoning $/1M-token: ${dollars_per_million(cost_r, tok_r):.2f}")
print(f"Normal $/1M-token:    ${dollars_per_million(cost_n, tok_n):.2f}")

# Năng lượng: reasoning ~80× chậm hơn → 80× nhiều năng lượng hơn
POWER_W, TOK_PER_SEC = 700, 50
wh_r = wh_per_query(reasoning_df["output_tokens"].mean(), POWER_W, TOK_PER_SEC / 80)
wh_n = wh_per_query(normal_df["output_tokens"].mean(), POWER_W, TOK_PER_SEC)
print(f"\nEnergy per reasoning query: {wh_r:.4f} Wh")
print(f"Energy per normal query:    {wh_n:.4f} Wh")
print(f"Ratio: {wh_r/wh_n:.1f}× more energy")

# Routing rule: nếu reasoning >10% traffic, đề xuất giảm
current_pct = len(reasoning_df) / len(df)
target_pct = 0.10
if current_pct > target_pct:
    excess = len(reasoning_df) - int(len(df) * target_pct)
    avg_r = cost_r / max(1, len(reasoning_df))
    avg_n = cost_n / max(1, len(normal_df))
    savings = (avg_r - avg_n) * excess
    print(f"\n=== Routing Rule Recommendation ===")
    print(f"Current reasoning: {current_pct:.1%} → Target: {target_pct:.1%}")
    print(f"Reroute {excess} requests/day to normal model")
    print(f"Potential savings: ${savings:.2f}/day")
    print(f"Rule: Use reasoning ONLY when task_complexity_score > 0.8")
```

---

### Extension 1 — Cải thiện `recommend_tier()`

**File sửa:** `finops/pricing.py` — hàm `recommend_tier()`

```python
# Interruption rate thực tế theo GPU type
INTERRUPT_RATES = {
    "H100": 0.02,  # H100 spot ít bị thu hồi (demand cao, supply ít)
    "A100": 0.05,
    "A10G": 0.10,  # A10G hay bị thu hồi
    "T4":   0.15,  # T4 rẻ nhất, dễ bị thu hồi nhất
}

def recommend_tier(hours_per_day, interruptible, reserved_discount=0.45,
                   gpu_type=None, job_days=None):
    """
    Phiên bản cải tiến: cân nhắc interruption rate và duration
    """
    duty = hours_per_day / 24.0
    be = break_even_utilization(reserved_discount)
    interrupt_rate = INTERRUPT_RATES.get(gpu_type, 0.05) if gpu_type else 0.05

    # Spot rất hấp dẫn khi: interruptible VÀ interrupt_rate thấp
    if interruptible and hours_per_day < 24:
        if interrupt_rate <= 0.05:
            return "spot_preferred"  # H100/A100 spot: ít rủi ro
        return "spot"

    # Job ngắn < 90 ngày: on-demand tốt hơn cam kết reserved
    if job_days and job_days < 90:
        return "on_demand"

    if duty >= be:
        # Phân biệt 1yr vs 3yr dựa trên duration
        if job_days and job_days >= 365 * 2:
            return "reserved_3yr"
        elif job_days and job_days >= 365:
            return "reserved_1yr"
        return "reserved"

    return "on_demand"
```

**Đo lường:** Chạy lại M3, so sánh `savings_pct` trước và sau khi sửa.

---

### Extension 2 — Right-sizing theo MBU

**File sửa:** `missions/m1_efficiency_audit.py` — thêm vào cuối

```python
import pandas as pd
from finops.metrics import compute_mbu

def right_size_recommendations(telemetry_df, catalog_df):
    """
    Với GPU memory-bound (MBU thấp), tìm GPU rẻ hơn có đủ băng thông.
    Tại sao không chọn GPU rẻ nhất theo $/GPU-hr?
    Vì GPU rẻ có thể thiếu băng thông → MBU cao hơn → hiệu quả kém hơn.
    """
    recs = []
    for gpu_id, group in telemetry_df.groupby("gpu_id"):
        gpu_type = group["gpu_type"].iloc[0]
        cat_row = catalog_df[catalog_df["gpu_type"] == gpu_type]
        if cat_row.empty:
            continue

        peak_bw = cat_row["peak_bw_tbs"].iloc[0]
        avg_achieved_bw = group["achieved_bw_tbs"].mean()
        avg_mbu = avg_achieved_bw / peak_bw
        current_price = cat_row["on_demand_hr"].iloc[0]

        if avg_mbu < 0.5:   # Memory-bound và sử dụng kém hiệu quả
            # Cần băng thông đủ cho 70% MBU target
            needed_bw = avg_achieved_bw / 0.7

            cheaper = catalog_df[
                (catalog_df["peak_bw_tbs"] >= needed_bw) &
                (catalog_df["on_demand_hr"] < current_price)
            ].sort_values("on_demand_hr")

            if len(cheaper) > 0:
                best = cheaper.iloc[0]
                savings_pct = (current_price - best["on_demand_hr"]) / current_price
                recs.append({
                    "gpu_id": gpu_id,
                    "current_type": gpu_type,
                    "recommended": best["gpu_type"],
                    "avg_mbu": f"{avg_mbu:.1%}",
                    "savings_pct": f"{savings_pct:.1%}",
                    "monthly_save": f"${(current_price - best['on_demand_hr'])*24*30:.0f}"
                })
    return pd.DataFrame(recs)

# Thêm vào cuối m1_efficiency_audit.py:
catalog_df = pd.read_csv("data/price_catalog.csv")
telemetry_df = pd.read_csv("data/gpu_telemetry.csv")
print("\n=== Extension 2: Right-sizing by MBU ===")
recs = right_size_recommendations(telemetry_df, catalog_df)
if len(recs) > 0:
    print(recs.to_string(index=False))
    total_monthly_save = sum(
        float(r.replace("$","")) for r in recs["monthly_save"]
    )
    print(f"\nTotal potential monthly savings: ${total_monthly_save:,.0f}")
else:
    print("No right-sizing opportunities found.")
```

---

### Extension 5 — Carbon-aware Scheduling

**Tạo file mới:** `missions/m5b_carbon_scheduling.py`

```python
"""Extension 5: Carbon-aware Scheduling for interruptible workloads"""
import pandas as pd
from finops.sustainability import REGION_CARBON, carbon_g, wh_per_query

workloads = pd.read_csv("data/workloads.csv")
interruptible = workloads[workloads["interruptible"] == 1]

REGION_INFO = {
    "europe-north1":   {"carbon": 20,   "price_kwh": 0.045},  # Na Uy - sạch nhất
    "us-west-2":       {"carbon": 150,  "price_kwh": 0.065},
    "us-east-1":       {"carbon": 400,  "price_kwh": 0.070},  # Mặc định hiện tại
    "asia-east1":      {"carbon": 520,  "price_kwh": 0.080},
    "europe-central2": {"carbon": 660,  "price_kwh": 0.075},  # Ba Lan - dơ nhất
}

POWER_W = 700
current_region = "us-east-1"
best_region = min(REGION_INFO, key=lambda r: REGION_INFO[r]["carbon"])

print("=== Extension 5: Carbon-aware Scheduling ===\n")
print(f"{'Region':<25} {'gCO2/kWh':>12} {'$/kWh':>8}")
print("-" * 50)
for r, info in sorted(REGION_INFO.items(), key=lambda x: x[1]["carbon"]):
    mark = " <- Best" if r == best_region else ""
    print(f"{r:<25} {info['carbon']:>12} {info['price_kwh']:>8.3f}{mark}")

# Tổng kWh tiêu thụ bởi interruptible jobs trong 1 tháng
total_hours = interruptible["hours_per_day"].sum() * 30
total_kwh = (POWER_W / 1000) * total_hours

carbon_cur  = total_kwh * REGION_INFO[current_region]["carbon"]
carbon_best = total_kwh * REGION_INFO[best_region]["carbon"]
cost_cur    = total_kwh * REGION_INFO[current_region]["price_kwh"]
cost_best   = total_kwh * REGION_INFO[best_region]["price_kwh"]

print(f"\nInterruptible jobs: {len(interruptible)} | Monthly: {total_hours:.0f}h = {total_kwh:.1f} kWh")
print(f"\nCurrent ({current_region}):")
print(f"  Carbon:      {carbon_cur/1000:.2f} kg CO2e")
print(f"  Energy cost: ${cost_cur:.2f}")
print(f"\nOptimized ({best_region}):")
print(f"  Carbon:      {carbon_best/1000:.2f} kg CO2e")
print(f"  Energy cost: ${cost_best:.2f}")
print(f"\nSavings:")
print(f"  Carbon reduced: {(carbon_cur-carbon_best)/1000:.2f} kg CO2e ({(1-carbon_best/carbon_cur):.1%})")
print(f"  Note: europe-north1 may have higher latency for Asian/US users")
```

---

## 12. Chuẩn bị nộp bài

### Checklist hoàn thành

```
[x] python data/generate.py       → 4 file CSV được tạo
[x] python missions/run_all.py    → M1-M5 chạy không lỗi
[x] python verify.py              → 11/11 checks passed
[x] pytest -q                     → 15 passed
[x] outputs/report.md tồn tại    → Có đủ 3 section bắt buộc
[x] outputs/savings.png tồn tại  → Biểu đồ waterfall 4 levers
[x] outputs/focus_export.csv     → FOCUS export từ M4
[x] >=2 extensions hoàn thành    → Có kết quả đo lường cụ thể
[x] write_up.md viết xong        → 1-2 trang phân tích
```

### Bài write-up ngắn (1–2 trang) — trả lời 5 câu hỏi

1. **Baseline vs. Optimized:**
   - Chi phí trước/sau ($/ngày và $/1M-token)?
   - Tổng savings bao nhiêu %?

2. **Phân tích từng đòn bẩy:**
   - Cascade, cache, batch mỗi cái đóng góp bao nhiêu?
   - Đòn bẩy nào mạnh nhất? Tại sao?

3. **GPU-Util Lie:**
   - GPU nào bị "lie"? MFU thực tế là bao nhiêu?
   - Tác động tài chính: đang trả bao nhiêu tiền cho FLOPs không dùng?

4. **Phần mở rộng:**
   - Extension nào đã làm?
   - Số liệu đo được cụ thể là gì?
   - Insight quan trọng nhất học được?

5. **Khuyến nghị cho NimbusAI (nếu bạn là FinOps Lead):**
   - Hành động 1 (ưu tiên cao nhất):
   - Hành động 2:
   - Hành động 3:

### File cần nộp

```
outputs/report.md           ← Báo cáo tự động từ M5
outputs/savings.png         ← Biểu đồ waterfall
outputs/focus_export.csv    ← FOCUS export từ M4
write_up.md (hoặc .pdf)    ← Bài viết phân tích tay
```

---

## 13. Bảng tra lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Venv chưa kích hoạt | Kích hoạt venv + `pip install -r requirements.txt` |
| `FileNotFoundError: data/gpu_telemetry.csv` | Chưa sinh dữ liệu | `python data/generate.py` |
| `verify.py fail M2 savings out of band` | Discount áp dụng sai | `cache_discount=0.10`, `batch_discount=0.50` |
| `pytest fail test_flag_util_lies` | Sai đơn vị so sánh | `util = float(r["gpu_util_pct"]) / 100.0` |
| `savings.png không được tạo` | matplotlib lỗi | `pip install matplotlib>=3.7` |
| `verify fail M4 coverage < 85%` | Data generate sai | Xóa CSV cũ + `python data/generate.py` |
| `TypeError` với `is_batch` | Pandas đọc thành string | `bool(int(row.get("is_batch", 0)))` |
| `verify pass nhưng pytest fail nhiều` | Engine sai, mission đúng | Kiểm tra lại `finops/*.py` |
| `No such file: outputs/` | Thư mục chưa tạo | `mkdir outputs` |

---

## 14. Bảng điểm & Tiêu chí chấm

| Phần | Điểm | Chi tiết |
|---|---|---|
| **A. verify.py** | 30 | 11/11=30, 10/11=25, 9/11=20, 8/11=15, 7/11=10, ≤6=5 |
| **B. pytest** | 20 | 15/15=20, 13-14=16, 10-12=12, 7-9=8, 4-6=4, ≤3=0 |
| **C. outputs/report.md** | 30 | Xem chi tiết bên dưới |
| **D. ≥2 Extensions** | 20 | Mỗi extension tối đa 10đ |
| **Tổng** | **100** | |

### Chi tiết chấm C (báo cáo kỹ thuật — 30 điểm)

| Tiêu chí | Điểm |
|---|---|
| Baseline spend + Optimized spend + % savings | 5 |
| Bảng breakdown từng lever với số tiền cụ thể | 5 |
| Sustainability: năng lượng/query, carbon, vùng tốt nhất | 5 |
| Giải thích cơ chế GPU-Util lie (không chỉ nêu kết quả) | 3 |
| Đề xuất hành động có độ ưu tiên rõ ràng | 4 |
| Nhận xét sustainability gắn với chi phí thực tế | 3 |
| savings.png có mặt và đúng 4 levers | 2 |
| Số liệu nhất quán giữa report.md và output missions | 3 |

### Thang điểm cho mỗi Extension (D)

| Điểm | Tiêu chí |
|---|---|
| **9–10** | Code chạy được + kết quả đo lường có số liệu + so sánh trước/sau rõ ràng + giải thích insight |
| **7–8** | Code chạy được + có kết quả đo lường, nhưng giải thích còn sơ sài |
| **5–6** | Code có lỗi nhỏ nhưng logic đúng, hoặc không có đo lường định lượng |
| **3–4** | Code chưa hoàn thiện nhưng hiểu đúng hướng |
| **0–2** | Chỉ viết comment ý định, không có code thực sự |

### Phân loại học lực

| Điểm | Phân loại |
|---|---|
| 90–100 | Xuất sắc — Pass tự động, test sạch, report insight sâu, ≥3 extensions |
| 80–89 | Tốt — Pass tự động, test sạch, report đủ, ≥2 extensions hoạt động |
| 70–79 | Khá — Gần pass, pytest >10/15, report đủ số liệu |
| 60–69 | Đạt — verify ≥8/11, pytest >7/15, report có cấu trúc |
| <60 | Cần cải thiện — Chưa hoàn thành flow cơ bản |

### Câu hỏi hay bị hỏi trong Oral Check

| Câu hỏi | Gợi ý trả lời |
|---|---|
| GPU-Util 98% có nghĩa GPU hiệu quả không? | Không — chỉ đo clock tick, không đo hiệu quả. MFU thấp do memory stall |
| Tại sao cần ≥80% tag coverage để chargeback? | 30% không tag → không biết ai dùng → tính phí không công bằng |
| $/GPU-hr vs $/1M-token khi nào trái ngược? | Cùng $/hr, team tối ưu phục vụ 10× token → $/1M-token khác 10× |
| LLM decode memory-bound, prefill compute-bound? | Decode: 1 token/lần, ít FLOP, nhiều HBM load. Prefill: song song nhiều FLOP |
| Spot phù hợp workload nào? | Training có checkpoint, batch offline. Không: real-time inference, cần SLA |

---

## 🚀 Quick Start — Chạy toàn bộ lab

```powershell
# 1. Vào thư mục dự án
cd "E:\26.AI-VIN\Track 02\Track2_Day25_2A202601414_NguyenNhatMinh"

# 2. Kích hoạt venv
.venv\Scripts\activate

# 3. Cài thư viện (lần đầu)
pip install -r requirements.txt

# 4. Sinh dữ liệu
python data/generate.py

# 5. Chạy tất cả missions
python missions/run_all.py

# 6. Kiểm tra toàn diện
python verify.py
pytest -q
```

**Output mong đợi cuối cùng:**
```
11/11 checks passed
15 passed in 0.42s
```

---

*Tổng hợp từ README.md, Guide.md và Rubric.md của Lab 25 — GPU FinOps Optimization*
*Cập nhật: 2026-08-27 | Nguyễn Nhật Minh — 2A202601414*
