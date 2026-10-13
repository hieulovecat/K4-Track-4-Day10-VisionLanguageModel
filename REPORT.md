Họ tên: <điền họ tên>
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Chạy trên Kaggle, 1 GPU Tesla T4 16 GB, QLoRA NF4 (ViT + projector + `lm_head` giữ fp16), 300 bước × (batch 2 × tích luỹ 4), 1 seed (42).
VRAM là `torch.cuda.max_memory_allocated()` (GiB) trong lúc **huấn luyện**, phút là thời gian **huấn luyện**. Riêng E0 không huấn luyện nên ghi VRAM đỉnh và thời gian **suy luận** 311 ảnh test.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 4.50 (suy luận) | 16.4 (suy luận) | 0.103 | 0.077–0.134 | 0.85 | 0.109 | 0.037 | 0.239 |
| E1 | LoRA LLM | 33,030,144 | 6.36 | 50.1 | 0.271 | 0.233–0.309 | 1.24 | 0.221 | 0.189 | 0.513 |
| E2 | + projector | 142,135,296 | 7.96 | 50.5 | 0.345 | 0.306–0.389 | 1.05 | 0.172 | 0.278 | 0.606 |
| E3 | + LoRA ViT | 148,426,752 | 8.13 | 59.0 | 0.433 | 0.395–0.474 | 0.89 | 0.298 | 0.370 | 0.647 |
| E4 | `coord_scale` = `norm1000` (chính là E3, chấm lại) | 148,426,752 | 8.13 | 59.0 | 0.433 | 0.395–0.474 | 0.89 | 0.298 | 0.370 | 0.647 |
| E4 | `coord_scale` = `pixel` (setup E3, nhãn + prompt + chấm theo pixel) | 148,426,752 | 8.12 | 58.6 | 0.427 | 0.384–0.472 | 0.90 | 0.314 | 0.342 | 0.676 |

Thêm: `parse_valid_rate` E0/E1/E2/E3/E4-pixel = 0.932 / 0.990 / 1.000 / 1.000 / 0.990; mean IoU của các cặp khớp = 0.612 / 0.678 / 0.685 / 0.701 / 0.681.
Thời gian suy luận 311 ảnh: E1 11.9, E2 9.9, E3 9.6, E4-pixel 12.0 phút; VRAM suy luận ≈ 4.4–4.7 GB.
Bootstrap **ghép cặp** (cùng tập ảnh, 2000 lần, `report/analysis/paired_bootstrap.json`):

| So sánh | ΔF1 | CI95 của ΔF1 |
|---|---|---|
| E1 − E0 | +0.168 | [+0.128, +0.207] |
| E2 − E1 | +0.074 | [+0.036, +0.110] |
| E3 − E2 | +0.088 | [+0.048, +0.127] |
| E3 − E1 | +0.162 | [+0.120, +0.207] |
| E4 pixel − E4 norm1000 (= E3) | −0.006 | [−0.048, +0.033] |

Ảnh minh hoạ trong `report/figures/` (xanh lá = box đúng, đỏ = dự đoán, chú thích ghi TP/FP/FN):
`e3_dung_0..2.jpg` (E3 đúng), `e3_sai_0..2.jpg` (E3 sai), `e0_0..1.jpg` (E0 trên cùng ảnh khó), `data_0..5.jpg` (nhãn gốc),
`roundtrip.jpg` (kiểm tra khứ hồi nhãn), `loss_curves.png` (loss E1–E3).

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm bằng cách dựng model trên `meta` device (không tải trọng số, nên số 4-bit không làm sai lệch), tổng **4,437.8M**:

| Khối | Tham số | Ghi chú |
|---|---|---|
| ViT (`model.visual`, 24 block + patch embed) | 306.2M | ảnh 640×360 → lưới 22×40 patch 16 px → 880 patch |
| Projector `visual.merger` | 27.3M | gộp 2×2 patch → **220 token** cho LLM (đã kiểm: `image_grid_thw = [1, 22, 40]`, đếm được 220 `<\|image_pad\|>`) |
| Projector `visual.deepstack_merger_list` (3 merger) | 81.8M | DeepStack: đặc trưng ở lớp 5, 11, 17 của ViT được chiếu và cộng vào các lớp đầu của LLM |
| LLM `model.language_model` (36 lớp) | 4,022.5M | trong đó `embed_tokens` 389.0M (151,936 × 2,560); `lm_head` **buộc chung** trọng số với embedding nên không có tham số riêng |

Khối được cập nhật (LoRA r = 16, α = 32):
- **E1**: chỉ LoRA trên LLM: `q,k,v,o,gate,up,down` × 36 lớp = **252 module**. Mỗi lớp 16·[(2560+4096)·2 + (2560+1024)·2 + (2560+9728)·3] = 917,504 → ×36 = **33,030,144** (khớp `trainable_params`). ViT và projector đóng băng.
- **E2**: E1 + **toàn bộ** projector (27.3M + 81.8M = **109,105,152**) → 142,135,296.
- **E3**: E2 + LoRA trên ViT `attn.qkv, attn.proj, mlp.linear_fc1, linear_fc2` × 24 block = 96 module (tổng **348**), thêm 24 × 16·[(1024+3072)+(1024+1024)+(1024+4096)·2] = **6,291,456** → 148,426,752.

Như vậy E2 chỉ thêm 0.8% tổng tham số nhưng là "cầu nối" giữa ảnh và ngôn ngữ; E3 thêm 6.3M tham số để chỉnh chính đặc trưng thị giác.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` = **0.932** (21/311 output hỏng). Model gốc theo đúng định dạng (0 output bọc ```` ```json ````, 0 chữ thừa, 0 nhãn lạ, 0 toạ độ > 1000) nên **thang toạ độ và lớp không phải lỗi chính**. Hai lỗi lớn nhất (`report/analysis/e0_error_stats.json`):

1. **Bỏ sót nghiêm trọng (recall 0.079)**: chỉ dự đoán 309 box cho 583 box đúng; 20 ảnh trả `[]` dù ảnh nào trong test cũng có hư hại. Box nhỏ gần như không thấy: recall pothole nhỏ (< 0.5% khung) là 1/68, crack 9/299. Ví dụ `vlcsnap-00201`: đúng là `[{"label":"manhole","bbox_2d":[856,725,919,789]},{"label":"crack","bbox_2d":[558,736,731,825]}]`, E0 trả `[]`.
2. **Bịa box / lặp vô hạn**: 185/309 box dự đoán là `crack` nhưng precision crack chỉ **0.046** (185 FP, trong đó 114 box không chạm box đúng nào, IoU < 0.1). Dạng nặng nhất là vòng lặp sinh box trượt dần theo trục y cho tới khi chạm `max_new_tokens` = 397, làm JSON bị cắt cụt → đó chính là 21 output hỏng. Ví dụ thô (`vlcsnap-2025-02-18-23h15m35s771`, thực tế chỉ có 1 manhole `[653,539,734,586]`):
   ```
   [{"label": "crack", "bbox_2d": [559, 375, 600, 407]}, {"label": "crack", "bbox_2d": [559, 398, 600, 425]},
    {"label": "crack", "bbox_2d": [559, 425, 600, 452]}, ... {"label": "crack", "bbox_2d": [559, 672, 600, 700]},
   ```
   (dừng ở 397 token, không có `]` đóng).

Ngoài ra box E0 khớp được cũng kém chính xác: mean IoU 0.612 so với 0.701 của E3.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

F1 **về đúng 0**:
- Dự đoán E1 (thang 0–1000) chấm với `--coord pixel`: F1 = **0.000** (TP 0, FP 537, FN 583) so với 0.271 khi chấm đúng (`report/analysis/e1_scored_as_pixel.json`).
- Ngược lại, dự đoán E4-pixel chấm với `--coord norm1000`: F1 = **0.000** (TP 0, FP 513) so với 0.427 khi chấm đúng (`report/analysis/e4_pixel_scored_as_norm1000.json`).

Vì sao: khi coi số 0–1000 là pixel, `score.py` nhân x với 1000/640 = 1.56 và y với 1000/360 = 2.78 trước khi so. Box bị đẩy về phía dưới-phải và kéo dài theo chiều dọc gần 3 lần; với box nhỏ (diện tích trung vị ≈ 0.4–2% khung hình) thì gần như không còn giao với box đúng, IoU < 0.5 với mọi cặp. Chiều ngược lại thì box co lại và dồn về góc trên-trái. Lỗi này **không làm hỏng JSON**: `parse_valid_rate` vẫn 0.99–1.0, chỉ có F1 sụp. Đây chính là lỗi im lặng nguy hiểm nhất.

Kiểm chứng: (1) chấm lại cùng file dự đoán với hai giá trị `--coord` (trên); (2) test đơn vị `test_wrong_scale_breaks_f1` trong `lab10/tests/test_labels.py`; (3) kiểm tra khứ hồi nhãn: YOLO → JSON 0–1000 số nguyên → pixel lệch tối đa ≈ 0.6 px, và chấm chính nhãn train bằng `score.py` cho F1 = 1.0 (`report/figures/roundtrip.jpg`). Lý do Qwen3-VL dùng 0–1000: toạ độ tương đối không phụ thuộc độ phân giải. Ở đây ảnh 640×360 thực ra được processor đổi thành 640×352 (bội của 32), nên "pixel" model thấy khác pixel của nhãn; thang tương đối miễn nhiễm với chuyện này.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Một ví dụ huấn luyện điển hình dài **405 token**: 346 token prompt (trong đó **220** token ảnh `<|image_pad|>` + system/prompt chữ) và chỉ **59** token đáp án (trung vị 14.6% chuỗi). Tôi đặt `labels = -100` cho toàn bộ tiền tố prompt (đã kiểm prompt là tiền tố đúng của chuỗi đầy đủ) và giải mã phần còn loss ra đúng `[{"label":...}]<|im_end|>`.

Nếu tính loss trên cả prompt và token ảnh:
- **Pha loãng tín hiệu**: loss là trung bình trên mọi token, nên đáp án chỉ chiếm ~15% gradient; ~85% còn lại dạy model chép lại một prompt **giống hệt nhau ở mọi ví dụ** và đoán chuỗi 220 token `<|image_pad|>` (cùng một id, dễ đoán tuyệt đối). Loss giảm nhanh trông "đẹp" nhưng không nói gì về khả năng định vị.
- Token ảnh không phải là "chữ" để dự đoán: nội dung ảnh nằm ở embedding được chèn vào vị trí placeholder, nên đích dự đoán là id placeholder là vô nghĩa và có thể làm lệch phân phối đầu ra (model có xu hướng sinh token đặc biệt).
- Thời gian/bộ nhớ không đổi (vẫn phải forward cả chuỗi), nên không có lợi gì đổi lại.

Hệ quả thực tế tôi thấy: chỉ với mask đúng, loss trên đáp án giảm từ ≈ 1.3 (bước 10) xuống 0.57/0.51/0.48 (E1/E2/E3, bước 300) và E1 vượt E0 +0.168 F1. Đáp án còn được kết thúc bằng `<|im_end|>` nên model học được khi nào dừng (chỉ 3/311 output E1 bị cắt).

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Projector (`merger`, `deepstack_merger_list`) nằm **bên trong** `model.visual`. Nếu khối visual bị lượng tử hoá, các `nn.Linear` của projector thành `bnb.Linear4bit`: trọng số là `Params4bit` dạng uint8 đóng gói (2 giá trị/byte) kèm hệ số scale theo block. Tensor nguyên không có gradient và optimizer không cập nhật được; muốn "train" phải giải lượng tử → cập nhật → lượng tử lại, mỗi bước mất độ chính xác (bước NF4 thô hơn rất nhiều so với cập nhật lr 2e-4). Vì vậy chỉ gắn được LoRA lên đó, không huấn luyện **đầy đủ** được.

Cách xử lý và một lỗi tôi gặp: GUIDE dùng `llm_int8_skip_modules=["visual","lm_head"]`, nhưng transformers 5.x so khớp tên module bằng `re.match` (neo ở đầu tên) hoặc `endswith`, nên `"visual"` **không khớp** `model.visual.blocks.0...` và ViT vẫn bị nén. Tôi thêm `"model.visual"` vào danh sách và viết `quantization_report()` để kiểm: số `Linear4bit` trong visual = **0**, trong LLM > 0 (notebook dừng lại nếu sai).

Float32: (1) huấn luyện fp16 dùng `GradScaler`, mà `unscale_` từ chối gradient fp16 ("Attempting to unscale FP16 gradients"); (2) fp16 chỉ có ~3 chữ số thập phân (ε ≈ 1e-3): cập nhật lr·grad cỡ 1e-5–1e-6 trên trọng số cỡ 1e-2 bị làm tròn mất, và trạng thái Adam (bình phương gradient) dễ tràn/triệt tiêu. Nên giữ **bản trọng số chủ fp32**, còn phép tính vẫn chạy fp16 trong `autocast`. Cái giá đo được: VRAM E2 − E1 = 7.96 − 6.36 = **1.60 GiB**, khớp ước tính 109.1M × (4 trọng số + 4 gradient + 8 Adam) byte = 1.63 GiB. Lúc suy luận tôi nạp lại projector đã lưu riêng (`projector.pt`, vì `save_pretrained` của peft chỉ lưu LoRA) và ép về fp16.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1,422 | 13 | 1,074 | 1,750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

(Đã kiểm: không phiên nào nằm ở hai split. Mỗi phiên trung vị 55 ảnh, lớn nhất 475 ảnh.)

Ảnh là frame cắt từ video dashcam, hai frame liền nhau gần như trùng nhau (cùng ổ gà, lệch vài pixel). Chia ngẫu nhiên thì hầu như mọi ảnh test đều có "anh em sinh đôi" trong train, nên model chỉ cần **nhớ** vị trí box của đoạn đường đó là đạt F1 cao. Con số khi đó đo khả năng ghi nhớ, không đo khả năng tổng quát hoá sang **đoạn đường mới**, là thứ ta cần khi triển khai. Chia theo phiên buộc test là những con đường model chưa từng thấy, và lộ ra cả lệch phân phối thật: crack trong test to hơn (diện tích trung vị 1.98% khung so với 1.35% ở train), pothole ít hơn hẳn (121 box). Tôi **không** chạy thêm thí nghiệm chia ngẫu nhiên nên không có số đo mức thổi phồng; nhận định trên dựa trên cấu trúc dữ liệu. Hệ quả phụ: test chỉ có 3 phiên, xem câu 8.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

| | F1 [CI95] | ΔF1 ghép cặp [CI95] | VRAM | Phút | FP/ảnh |
|---|---|---|---|---|---|
| E1 | 0.271 [0.233, 0.309] | | 6.36 | 50.1 | 1.24 |
| E2 | 0.345 [0.306, 0.389] | E2−E1 +0.074 [+0.036, +0.110] | 7.96 (+1.60) | 50.5 (+0.4) | 1.05 |
| E3 | 0.433 [0.395, 0.474] | E3−E2 +0.088 [+0.048, +0.127] | 8.13 (+0.18) | 59.0 (+8.5) | 0.89 |

**Đáng.** Mỗi bước mở thêm đều tăng F1 với CI95 của hiệu không chứa 0 (trong 2000 lần bootstrap, E2 và E3 đều tốt hơn ở 100% lần). E3 so với E1: +0.162 F1 (+60% tương đối) đổi lấy +1.8 GB VRAM và +18% thời gian, vẫn vừa thoải mái trong T4 16 GB.
- Mở projector (E2) tốn VRAM nhất (do 109M tham số fp32 + Adam, câu 5) nhưng gần như không thêm thời gian (không có gradient đi qua ViT). Tăng chủ yếu ở crack (0.189 → 0.278) và manhole (0.513 → 0.606), precision tổng 0.283 → 0.368.
- LoRA ViT (E3) chỉ thêm 6.3M tham số nhưng cho bước nhảy lớn nhất: gradient phải đi ngược qua 24 block ViT (thêm thời gian, gradient checkpointing nên VRAM tăng ít). Đặc trưng thị giác gốc được huấn luyện cho ảnh tổng quát; để nhìn ra ổ gà/vết nứt nhỏ trên mặt đường cần chỉnh chính bộ mã hoá ảnh. Recall tăng 0.261 → 0.326 → 0.408, mean IoU 0.678 → 0.701.
- Loss cuối (đáp án): 0.566 / 0.514 / 0.485 — cùng thứ tự với F1 test.

Lưu ý: chỉ 1 seed mỗi setup; CI bootstrap chỉ phản ánh độ biến thiên do chọn ảnh test, không phản ánh độ biến thiên do khởi tạo/thứ tự dữ liệu khi huấn luyện.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 bootstrap (theo ảnh, 1000 lần) rộng khoảng **±0.04 F1**: E1 0.233–0.309 (rộng 0.076), E2 0.306–0.389 (0.083), E3 0.395–0.474 (0.079). Hệ quả:
- **Đứng vững**: E0 < E1 < E2 < E3. Chênh lệch giữa các bước (0.07–0.17) lớn hơn độ bất định. Nếu chỉ nhìn hai CI rời thì E1 (≤ 0.309) và E2 (≥ 0.306) còn chồng nhau, nhưng bootstrap **ghép cặp** (lấy mẫu lại cùng tập ảnh cho cả hai setup, khử phần "ảnh khó/dễ" chung) cho CI của hiệu [+0.036, +0.110], không chứa 0. Với so sánh trên cùng test set, nên dùng cách ghép cặp.
- **Không kết luận được**: E4 (ΔF1 −0.006, CI [−0.048, +0.033]); chênh lệch F1 từng lớp có ít box. Ví dụ pothole "giảm" 0.221 → 0.172 từ E1 sang E2 dựa trên đúng 19 TP mỗi bên trong 121 box, là nhiễu chứ không phải projector làm hại pothole (E3 lại lên 0.298).
- **CI thực tế còn rộng hơn**: bootstrap giả định 311 ảnh độc lập, nhưng test chỉ gồm **3 phiên quay**; các frame trong một phiên tương quan mạnh (cùng đoạn đường, cùng ánh sáng), nên số mẫu hiệu dụng nhỏ hơn 311 nhiều. Kết luận "E3 tốt hơn" là cho 3 đoạn đường này; tổng quát hoá sang đường khác cần thêm phiên. Thêm vào đó là phương sai giữa các seed (không đo).

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở setup tốt nhất E3: **pothole kém nhất (F1 0.298)**, rồi crack (0.370), manhole tốt nhất (0.647). Phân rã lỗi trong `report/analysis/error_breakdown.json`:

| E3 | Precision | Recall | Recall box nhỏ (< 0.5% khung) | FP: sai lớp / lệch vị trí (0.1 ≤ IoU < 0.5) / bịa (IoU < 0.1) |
|---|---|---|---|---|
| pothole | 0.340 | 0.264 | 14/68 = 21% | 23 / 10 / 29 |
| crack | 0.384 | 0.358 | 0/15 | 21 / 91 / 60 |
| manhole | 0.692 | 0.607 | 25/64 = 39% | 5 / 14 / 25 |

- **Pothole — kích thước + ngữ nghĩa + số mẫu test**: box rất nhỏ (diện tích trung vị 0.42% khung, 68/121 box < 0.5%), nên một ổ gà chỉ phủ vài token trong lưới 22×40 mà ViT nhìn thấy. Ranh giới với crack mơ hồ: ổ gà thường nằm giữa vùng nứt, và nhãn có khi lồng box pothole trong box crack lớn (`report/figures/e3_sai_0.jpg`, `e3_sai_2.jpg`), nên 23 FP là **gọi nhầm lớp**. Train có nhiều pothole (1,074) nhưng test chỉ 121 box từ 3 phiên nên số đo pothole dao động mạnh nhất.
- **Crack — định vị mơ hồ**: crack là vật thể dài, mảnh, không có ranh giới rõ; một vết nứt có thể được gán bằng một box dài hoặc vài box ngắn. Hơn nửa FP (91/172) là box đã chạm đúng vết nứt nhưng IoU < 0.5, tức sai quy ước khoanh vùng chứ không hẳn là "không thấy". Đây cũng là lớp E0 bịa nhiều nhất (precision 0.046).
- **Manhole — dễ nhất**: hình dạng cứng (chữ nhật/tròn), viền và kết cấu kim loại khác biệt rõ, nhãn nhất quán. Box vừa (0.5–2% khung) được tìm thấy 72/96.

Ảnh minh hoạ: đúng — `e3_dung_0.jpg` (crack + manhole + pothole, TP 3/3), `e3_dung_1.jpg` (cụm pothole, TP 3, FN 1), `e3_dung_2.jpg` (2 manhole); sai — `e3_sai_0.jpg` (đường nhiều ổ gà lồng trong vùng nứt, TP 1, FP 4, FN 5), `e3_sai_1.jpg` (đường phố: bịa crack trên mặt nhựa trơn, bỏ sót manhole sát mép phải; TP 1, FP 3, FN 4), `e3_sai_2.jpg` (một box crack lớn bao các pothole; model khoanh nhiều box nhỏ lệch lớp/vị trí, TP 0); so sánh `e0_0.jpg` (E0 trên cùng ảnh với `e3_sai_0`, TP 0).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Biến: **`coord_scale`**. Giữ nguyên setup tốt nhất E3 (r = 16, projector, LoRA ViT, 300 bước, seed 42), chỉ đổi thang toạ độ đồng thời ở **nhãn huấn luyện, prompt và lúc chấm** (`--coord pixel`). Giá trị `norm1000` chính là lần chạy E3 (chấm lại, không huấn luyện lại).

| | F1 [CI95] | Precision | Recall | mean IoU | parse | F1 pothole / crack / manhole |
|---|---|---|---|---|---|---|
| norm1000 | 0.433 [0.395, 0.474] | 0.461 | 0.408 | 0.701 | 1.000 | 0.298 / 0.370 / 0.647 |
| pixel | 0.427 [0.384, 0.472] | 0.456 | 0.401 | 0.681 | 0.990 | 0.314 / 0.342 / 0.676 |

ΔF1 (pixel − norm1000) = **−0.006, CI95 ghép cặp [−0.048, +0.033]**, tức **không có khác biệt có ý nghĩa**. Kết luận đáng tin là: khi nhất quán từ đầu đến cuối, model học thang pixel gần như tốt bằng thang 0–1000 sau 300 bước (ảnh có kích thước cố định 640×360 nên hai thang chỉ khác một phép co giãn tuyến tính). Điều quyết định là **sự nhất quán**: lệch thang giữa huấn luyện và chấm làm F1 về 0 (câu 3). Thang 0–1000 nhỉnh hơn ở mean IoU (0.701 so với 0.681) và JSON hợp lệ (1.000 so với 0.990), và vẫn là lựa chọn đúng khi ảnh có nhiều kích thước hoặc bị resize (ở đây 360 → 352), nhưng với 1 seed và CI như trên tôi không khẳng định thang nào tốt hơn.

## Nhìn lại

- **Lỗi dữ liệu công cụ chấm không bắt**: ảnh train `vlcsnap-00058` có một box crack cao **0** (`yolo = [0.639062, 0.786111, 0.003125, 0.0]`). Đổi sang số nguyên thành `[637,786,641,786]` (y1 = y2), và `score.py` coi **cả JSON của ảnh** là hỏng nếu có một box như vậy. Nếu không sửa, model sẽ học sinh box suy biến và tự làm tụt `parse_valid_rate`. Tôi nới box suy biến thành 1 đơn vị và thêm kiểm tra "mọi đáp án đều được `score.py` đọc hợp lệ" trước khi huấn luyện. `prepare_data.py` chỉ kiểm giá trị trong [0, 1] nên không chặn w = 0 hay h = 0.
- **Lỗi thiết lập trong hướng dẫn**: `llm_int8_skip_modules=["visual", ...]` không còn khớp tên module trong transformers 5.x (câu 5), nên phải tự kiểm số `Linear4bit` trong ViT; nếu không kiểm, E2/E3 sẽ âm thầm huấn luyện trên projector bị nén. Trên Kaggle còn gặp `torchao` 0.10 cài sẵn khiến peft 0.21 báo lỗi khi tạo LoRA (đã gỡ `torchao`).
- **Lỗi mô hình mà điểm số che đi**: các output hỏng (E0 21, E1 3, E4-pixel 3) đều là **vòng lặp** sinh box gần trùng nhau cho tới khi chạm `max_new_tokens` = 397. `score.py` chỉ phạt box trùng như FP và tính cả ảnh là hỏng khi bị cắt; một bước khử trùng lặp (NMS trên JSON) hoặc `repetition_penalty` lúc sinh có thể sửa rẻ, nhưng tôi chưa thử để giữ so sánh sạch.
- **Điều tôi chưa chắc**: CI bootstrap theo ảnh bỏ qua tương quan trong phiên (test chỉ 3 phiên), và tôi chưa đo phương sai giữa các seed. Muốn khẳng định chắc hơn cần bootstrap theo phiên hoặc thêm seed, nhưng tốn GPU.
- Chưa làm E5 (GRPO): hàm reward `reward_iou` và test đã có trong `lab10/`, nhưng chưa chạy.
