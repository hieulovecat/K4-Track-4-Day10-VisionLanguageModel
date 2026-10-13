"""Sinh lab_day10.ipynb (bản đã điền mã, chạy trên Kaggle hoặc Colab). Chạy: python lab10/build_notebook.py"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường

Notebook chạy được trên **Kaggle** (GPU T4) hoặc **Colab** (T4). Mã chính nằm trong package `lab10/`:

| File | Nội dung |
|---|---|
| `lab10/labels.py` | YOLO → JSON 0–1000, prompt, reward IoU (Python thuần, có test) |
| `lab10/vlm.py` | nạp model 4-bit, `encode_example` + mask loss, `collate`, `apply_setup`, vòng train, suy luận, GRPO |
| `lab10/experiments.py` | điều phối train → suy luận → chấm, **chạy tiếp được** nếu phiên bị ngắt |
| `lab10/analysis.py` | phân tích lỗi, vẽ box, bootstrap ghép cặp |

**Cách chạy trên Kaggle:** bật *Accelerator = GPU T4 x2*, *Internet = On*, thêm dataset
`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes` qua *Add Input*.
Chạy thử với `SMOKE = True` (≈ 20 phút), sau đó đặt `SMOKE = False` rồi dùng **Save Version → Save & Run All (Commit)**
để chạy toàn bộ ở chế độ nền (≈ 4–5 giờ).
"""),
    md("## 0. Môi trường (Kaggle / Colab, GPU T4)"),
    code("""
# ===================== CẤU HÌNH =====================
REPO_URL    = "https://github.com/hieulovecat/K4-Track-4-Day10-VisionLanguageModel.git"
SMOKE       = True            # True: chạy thử nhanh mọi bước (vài bước train, 8 ảnh val). False: chạy thật.
E4_VARIABLE = "coord_scale"   # "coord_scale" | "rank" | "data_size"  (chọn ĐÚNG MỘT biến)
RUN_E5      = False           # bonus GRPO (+≈ 1 giờ)
INFER_BS    = 8               # batch suy luận; giảm về 4 nếu hết VRAM
PREV_OUTPUT = None            # vd "/kaggle/input/lab10-run1": output lần chạy trước để chạy tiếp
# ====================================================
"""),
    code("""
import os, sys, subprocess, shutil, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # Kaggle T4 x2: chỉ dùng 1 GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ON_KAGGLE = Path("/kaggle/working").exists()

if Path("scripts/score.py").exists():                   # đang đứng trong repo (local / đã clone)
    ROOT = Path.cwd()
else:
    ROOT = Path("/kaggle/working/repo" if ON_KAGGLE else "/content/repo")
    if not (ROOT / "scripts/score.py").exists():
        # Nếu bạn upload repo thành Kaggle Dataset (repo chưa public) thì dùng bản đó, không thì git clone
        uploaded = sorted(Path("/kaggle/input").glob("**/lab10/vlm.py")) if ON_KAGGLE else []
        if uploaded:
            shutil.copytree(uploaded[0].parents[1], ROOT,
                            ignore=shutil.ignore_patterns("data", "work", ".git"))
        else:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("ROOT =", ROOT, "| Kaggle" if ON_KAGGLE else "")
"""),
    code("""
!nvidia-smi -L
!pip -q install -r requirements-colab.txt
# Kaggle cài sẵn torchao 0.10; peft 0.21 thấy torchao < 0.16 là báo ImportError khi tạo LoRA.
# Lab không dùng torchao nên gỡ đi (sau đó nhớ Restart kernel nếu đã import peft/transformers).
!pip -q uninstall -y torchao
"""),
    code("""
def get_secret(name):
    \"\"\"Đọc khoá từ Kaggle Secrets / Colab Secrets. KHÔNG ghi khoá vào notebook.\"\"\"
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

_hf = get_secret("HF_TOKEN")        # tuỳ chọn: tải model nhanh hơn
if _hf:
    os.environ["HF_TOKEN"] = _hf
"""),
    code("""
!python -m pytest -q
!python -m pytest -q lab10/tests
"""),
    code("""
# Chạy tiếp từ output của lần chạy trước (Add Input → Notebook output của chính bạn)
if PREV_OUTPUT:
    prev = Path(PREV_OUTPUT)
    src = prev / "repo" if (prev / "repo").exists() else prev
    for sub in ("report/evidence", "report/figures", "report/analysis", "work"):
        if (src / sub).exists():
            shutil.copytree(src / sub, ROOT / sub, dirs_exist_ok=True)
            print("khôi phục", sub)
"""),
    md("## 1. Dữ liệu: tải và chia split"),
    code("""
def find_dataset():
    base = Path("/kaggle/input")
    if base.exists():
        for p in sorted(base.rglob("labels-YOLO")):
            if (p.parent / "images").is_dir():
                return p.parent
    return None

if not Path("data/splits/meta.json").exists():
    src = find_dataset()
    if src:                                   # Kaggle: dataset đã gắn qua Add Input
        !python scripts/prepare_data.py --source "{src}"
    else:                                     # Colab: tải bằng Kaggle API (Secrets)
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            os.environ[k] = get_secret(k) or ""
        !python scripts/prepare_data.py --kaggle
"""),
    code("""
from collections import Counter
from scripts.labcommon import read_jsonl, CLASSES

splits = {s: read_jsonl(f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
META = json.loads(Path("data/splits/meta.json").read_text())
print("dataset_version:", META["dataset_version"])
for s, rows in splits.items():
    c = Counter(o["label"] for r in rows for o in r["objects"])
    print(f"{s:5s} {len(rows):5d} ảnh  {sum(c.values()):5d} box  " + "  ".join(f"{k}={c[k]}" for k in CLASSES)
          + f"  phiên={len({r['session'] for r in rows})}")
sess = {s: {r["session"] for r in rows} for s, rows in splits.items()}
assert not (sess["train"] & sess["val"] or sess["train"] & sess["test"] or sess["val"] & sess["test"])
print("OK: không phiên nào nằm ở hai split")
"""),
    code("""
# Vẽ 6 ảnh train với box đúng (xanh) để hiểu nhãn
from IPython.display import display
from lab10 import analysis
from lab10.labels import entry_answer
import random

FIG = Path("report/figures"); FIG.mkdir(parents=True, exist_ok=True)
for i, e in enumerate(random.Random(0).sample([r for r in splits["train"] if r["objects"]], 6)):
    p = analysis.draw_example(e, "[]", FIG / f"data_{i}.jpg", title="ground truth")
    if i < 2:
        display(analysis.Image.open(p))
"""),
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code("""
import inspect
from lab10 import labels
print(inspect.getsource(labels.yolo_to_box)); print(inspect.getsource(labels.objects_to_answer))
e = next(r for r in splits["train"] if len(r["objects"]) >= 3)
print("YOLO :", [(o["label"], [round(v, 4) for v in o["yolo"]]) for o in e["objects"]])
print("JSON :", entry_answer(e))
print("pixel:", entry_answer(e, "pixel"))
print("\\nPROMPT:", labels.PROMPT_NORM1000)
"""),
    code("""
# Kiểm tra khứ hồi: JSON 0–1000 (số nguyên) -> pixel phải trùng box gốc (sai số làm tròn <= 1 px)
from scripts.score import evaluate, parse_output
ALL = splits["train"] + splits["val"] + splits["test"]
thin = [(e["image_id"], o["yolo"]) for e in ALL for o in e["objects"]
        if round(o["yolo"][2] * 1000) < 1 or round(o["yolo"][3] * 1000) < 1]
print(f"box mảnh < 1/1000 khung (đã nới thành 1 đơn vị): {len(thin)}", thin[:3])
bad = [e["image_id"] for e in ALL for c in ("norm1000", "pixel") if parse_output(entry_answer(e, c)) is None]
assert not bad, f"đáp án bị score.py coi là JSON hỏng: {bad[:5]}"
worst = 0.0
for e in splits["train"]:
    pr = [labels.box_1000_to_pixel(b["bbox"], e["width"], e["height"]) for b in parse_output(entry_answer(e))]
    for o in e["objects"]:
        xc, yc, w, h = o["yolo"]
        gt_px = (max(0, (xc - w / 2)) * e["width"], max(0, (yc - h / 2)) * e["height"],
                 min(1, (xc + w / 2)) * e["width"], min(1, (yc + h / 2)) * e["height"])
        worst = max(worst, min(max(abs(a - c) for a, c in zip(gt_px, p)) for p in pr))
print(f"sai số khứ hồi lớn nhất: {worst:.2f} px")
r = evaluate(splits["train"], {e["image_id"]: entry_answer(e) for e in splits["train"]}, n_boot=10)
print("chấm chính nhãn train bằng score.py: F1 =", r["micro"]["f1"], " meanIoU =", round(r["mean_iou"], 4))
e = splits["train"][0]
display(analysis.Image.open(analysis.draw_example(e, entry_answer(e), FIG / "roundtrip.jpg", title="round-trip: GT drawn as prediction")))
"""),
    md("## 3. E0: zero-shot"),
    code("""
import torch
from lab10 import vlm, experiments
from lab10.labels import get_prompt

SPLIT = "val" if SMOKE else "test"
EVAL = splits["val"][:8] if SMOKE else splits["test"]
EVID = Path("work/smoke_evidence") if SMOKE else Path("report/evidence")
WORK = Path("work/smoke") if SMOKE else Path("work")

t0 = time.time()
model, processor = vlm.load_base()
print(f"nạp model {time.time() - t0:.0f}s,  VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
q = vlm.quantization_report(model)
print(q)
assert q["visual_4bit"] == 0 and q["llm_4bit"] > 0, "ViT bị lượng tử hoá 4-bit: kiểm tra llm_int8_skip_modules"
"""),
    code("""
# Câu 1: số tham số từng khối (tính trên meta device) + số module LoRA khớp regex
pb = vlm.param_breakdown()
for k, v in pb.items():
    print(f"{k:28s} {v / 1e6:10.1f}M" if "modules" not in k else f"{k:28s} {v}")
"""),
    code("""
# Độ dài đáp án dài nhất (token) -> max_new_tokens đủ để không cắt cụt JSON
ans_tok = {c: vlm.max_answer_tokens(processor, splits["train"] + splits["test"], c) for c in ("norm1000", "pixel")}
MAX_NEW = max(256, max(ans_tok.values()) + 32)
print("token đáp án dài nhất:", ans_tok, "-> max_new_tokens =", MAX_NEW)

lab = experiments.Lab(splits, eval_entries=EVAL, split=SPLIT, evid_dir=EVID, work_dir=WORK,
                      infer_bs=INFER_BS, max_new_tokens=MAX_NEW,
                      overrides={"max_steps": 4} if SMOKE else None)
ev0 = lab.zero_shot(model, processor)
"""),
    code("""
# Câu 2: output thô và thống kê lỗi của E0
preds0 = {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path("e0"))}
for e in EVAL[:5]:
    print("─" * 100, "\\nGT  :", entry_answer(e), "\\nRAW :", preds0[e["image_id"]][:400])
st0 = analysis.raw_error_stats(EVAL, preds0)
print(json.dumps(st0, indent=1, ensure_ascii=False))
(Path("report/analysis")).mkdir(parents=True, exist_ok=True)
if not SMOKE:
    Path("report/analysis/e0_error_stats.json").write_text(json.dumps(st0, indent=1, ensure_ascii=False))
"""),
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code("""
print(inspect.getsource(vlm.encode_example)); print(inspect.getsource(vlm.collate))
e = next(r for r in splits["train"] if r["objects"])
enc = vlm.encode_example(processor, vlm.load_image(e), get_prompt(e), entry_answer(e))
n_img = vlm.image_token_count(processor, enc)
lab_ids = enc["labels"][0]
print({k: tuple(v.shape) for k, v in enc.items()})
print("token ảnh:", n_img, "| image_grid_thw:", enc["image_grid_thw"].tolist())
print("số token tính loss:", int((lab_ids != -100).sum()), "/", lab_ids.numel())
print("giải mã phần có loss:", repr(processor.tokenizer.decode(lab_ids[lab_ids != -100])))
assert n_img == 220
b = vlm.collate([enc, vlm.encode_example(processor, vlm.load_image(splits["train"][1]), get_prompt(splits["train"][1]), "[]")],
                processor.tokenizer.pad_token_id)
print("batch:", {k: tuple(v.shape) for k, v in b.items()})
del model; vlm.free_memory()
"""),
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code("""
print(inspect.getsource(vlm.apply_setup))
print("Regex LLM:", vlm.LLM_TARGETS)
print("Regex ViT:", vlm.VIT_TARGETS)
print("module khớp: E1/E2 =", pb["lora_modules_llm"], "| E3 =", pb["lora_modules_llm"] + pb["lora_modules_vit"])
assert pb["lora_modules_llm"] == 252 and pb["lora_modules_llm"] + pb["lora_modules_vit"] == 348
for k, c in experiments.CONFIGS.items():
    print(k, lab.cfg(c))
"""),
    md("""
## 6. Vòng huấn luyện

Mỗi setup: AdamW lr 2e-4, warmup 5% + cosine, fp16 + GradScaler, gradient checkpointing, batch 2 × tích luỹ 4, 300 bước.
Adapter (+ projector với E2/E3) được lưu mỗi 100 bước vào `work/runs/<eN>/`; `run.json` ghi
`trainable_params`, `peak_vram_gb`, `train_minutes` và log loss. Chạy lại cell này sẽ **bỏ qua** setup đã xong.
"""),
    code("""
RUNS = {}
for key in ("e1", "e2", "e3"):
    print("=" * 30, key, "=" * 30)
    RUNS[key] = lab.train(experiments.CONFIGS[key])
"""),
    code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.5))
for key, run in RUNS.items():
    ax.plot([r["step"] for r in run["loss_log"]], [r["loss"] for r in run["loss_log"]], label=key.upper())
ax.set_xlabel("bước tối ưu"); ax.set_ylabel("loss (trên token đáp án)"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=120); plt.show()
"""),
    md("## 7. Suy luận và chấm E1–E3"),
    code("""
EV = {"e0": ev0}
for key in ("e1", "e2", "e3"):
    EV[key] = lab.evaluate(experiments.CONFIGS[key])
"""),
    code("""
from IPython.display import Markdown
print(analysis.markdown_table([EVID / f"{k}.json" for k in ("e0", "e1", "e2", "e3")]))
display(Markdown(analysis.markdown_table([EVID / f"{k}.json" for k in ("e0", "e1", "e2", "e3")])))
print("E1 - E0 =", round(EV["e1"]["micro"]["f1"] - EV["e0"]["micro"]["f1"], 3), "(cần >= 0.10)")
"""),
    code("""
# Câu 7–8: bootstrap GHÉP CẶP (cùng tập ảnh) cho hiệu F1 giữa các setup
P = {k: {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path(k))} for k in ("e0", "e1", "e2", "e3")}
PAIRS = {}
for a, b in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3")):
    PAIRS[f"{b}-{a}"] = pb_ = analysis.paired_bootstrap(EVAL, P[a], P[b])
    print(f"F1({b}) - F1({a}) = {pb_['diff']:+.3f}  CI95 [{pb_['ci95'][0]:+.3f}, {pb_['ci95'][1]:+.3f}]  "
          f"P({b} tốt hơn) = {pb_['p_b_better']:.2f}")
if not SMOKE:
    Path("report/analysis/paired_bootstrap.json").write_text(json.dumps(PAIRS, indent=1))
"""),
    code("""
# Câu 3: chấm dự đoán E1 (thang 0–1000) như thể là pixel -> F1 sụp
ev_wrong = lab.rescore("e1", Path("report/analysis") / ("e1_scored_as_pixel.json" if not SMOKE else "smoke_e1_px.json"),
                       "E1_scored_as_pixel", coord="pixel")
"""),
    code("""
# Câu 9 + ảnh minh hoạ: setup tốt nhất, 3 ảnh đúng + 3 ảnh sai; E0 thêm 2 ảnh
BEST = max(("e1", "e2", "e3"), key=lambda k: EV[k]["micro"]["f1"])
print("setup tốt nhất:", BEST)
good, bad = analysis.pick_examples(EVAL, P[BEST], n_good=3, n_bad=3)
by_id = {e["image_id"]: e for e in EVAL}
shown = []
for tag, ids in (("dung", good), ("sai", bad)):
    for i, iid in enumerate(ids):
        shown.append(analysis.draw_example(by_id[iid], P[BEST][iid], FIG / f"{BEST}_{tag}_{i}.jpg",
                                           title=f"{BEST.upper()} ({'good' if tag == 'dung' else 'bad'})"))
for i, iid in enumerate(bad[:2]):
    shown.append(analysis.draw_example(by_id[iid], P["e0"][iid], FIG / f"e0_{i}.jpg", title="E0 zero-shot"))
for p in shown[:4]:
    display(analysis.Image.open(p))
"""),
    md("## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)"),
    code("""
# E4: đổi ĐÚNG MỘT biến so với setup tốt nhất; giá trị trùng setup gốc thì chấm lại dự đoán sẵn có
base = experiments.CONFIGS[BEST]
if E4_VARIABLE == "coord_scale":
    E4 = [("pixel", {**base, "key": "e4_coord_scale_pixel", "name": "E4_coord_scale_pixel", "coord": "pixel"})]
    E4_REUSE = ("norm1000", "e4_coord_scale_norm1000")
elif E4_VARIABLE == "rank":
    E4 = [(r, {**base, "key": f"e4_rank_{r}", "name": f"E4_rank_{r}", "r": r, "alpha": 2 * r}) for r in (4, 64)]
    E4_REUSE = None
elif E4_VARIABLE == "data_size":
    E4 = [(0.1, {**base, "key": "e4_data_size_0.1", "name": "E4_data_size_0.1", "train_fraction": 0.1})]
    E4_REUSE = (1.0, "e4_data_size_1.0")
else:
    raise ValueError(E4_VARIABLE)

for val, cfg in E4:
    print("=" * 30, cfg["key"], "=" * 30)
    EV[cfg["key"]] = lab.run(cfg)
if E4_REUSE:
    val, k = E4_REUSE
    EV[k] = lab.rescore(BEST, EVID / f"{k}.json", k.upper(), extra_meta={"same_run_as": BEST})
"""),
    code("""
e4_keys = sorted(k for k in EV if k.startswith("e4"))
display(Markdown(analysis.markdown_table([EVID / f"{k}.json" for k in [BEST] + e4_keys])))
if E4_VARIABLE == "coord_scale":   # Câu 3: ngược lại, dự đoán pixel mà chấm như 0–1000
    lab.rescore("e4_coord_scale_pixel", Path("report/analysis") / ("e4_pixel_scored_as_norm1000.json" if not SMOKE else "smoke_px_1000.json"),
                "E4_pixel_scored_as_norm1000", coord="norm1000")
if len(e4_keys) == 2 and E4_REUSE is None:
    a, b = e4_keys
    print(analysis.paired_bootstrap(EVAL, {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path(a))},
                                    {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path(b))}))
elif E4_REUSE:
    k_new = E4[0][1]["key"]
    pn = {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path(k_new))}
    print(f"{k_new} - {BEST}:", analysis.paired_bootstrap(EVAL, P[BEST], pn, coord_b=E4[0][1]["coord"]))
"""),
    md("""
## 9. E5 (tuỳ chọn, +10): GRPO với reward IoU

`lab10/labels.py::reward_iou` có test trong `lab10/tests/test_labels.py` (hoàn hảo, JSON hỏng, sai lớp, thừa box,
rỗng, box khổng lồ). Vòng GRPO tự viết trong `vlm.grpo_train` (on-policy, beta = 0, 4 câu trả lời/ảnh,
tối đa 128 token, 60 bước, lr 5e-6), khởi tạo từ adapter tốt nhất.
"""),
    code("""
if RUN_E5:
    from lab10.labels import reward_iou
    print(inspect.getsource(reward_iou))
    e5_dir = WORK / "runs" / "e5"
    if not (e5_dir / "run.json").exists():
        model, processor = vlm.load_base()
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = vlm.attach_trained(model, lab.run_dir(BEST), is_trainable=True)
        run5 = vlm.grpo_train(model, processor, splits["train"], e5_dir, steps=3 if SMOKE else 60,
                              init_dir=lab.run_dir(BEST))
        del model; vlm.free_memory()
    run5 = json.loads((e5_dir / "run.json").read_text())
    cfg5 = {**experiments.CONFIGS[BEST], "key": "e5", "name": "E5_GRPO"}
    EV["e5"] = lab.evaluate(cfg5)
    P["e5"] = {r["image_id"]: r["raw"] for r in read_jsonl(lab.pred_path("e5"))}
    print("E5 - SFT:", analysis.paired_bootstrap(EVAL, P[BEST], P["e5"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([r["reward_mean"] for r in run5["log"]]); ax[0].set_title("reward trung bình")
    ax[1].plot([r["frac_empty"] for r in run5["log"]], label="tỉ lệ []")
    ax[1].plot([r["mean_box_area"] for r in run5["log"]], label="diện tích box TB (tỉ lệ khung)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "grpo_log.png", dpi=120); plt.show()
"""),
    md("## 10. Tổng hợp và đóng gói kết quả"),
    code("""
keys = [k for k in ("e0", "e1", "e2", "e3") if (EVID / f"{k}.json").exists()] + sorted(p.stem for p in EVID.glob("e4*.json"))
keys += ["e5"] if (EVID / "e5.json").exists() else []
table = analysis.markdown_table([EVID / f"{k}.json" for k in keys])
display(Markdown(table))
if not SMOKE:
    Path("report/summary_table.md").write_text(table + "\\n", encoding="utf-8")
    print("dataset_version:", META["dataset_version"])
    !python scripts/check_submission.py
"""),
    code("""
# Đóng gói những gì cần commit (KHÔNG có trọng số / dữ liệu) -> tải về từ tab Output của Kaggle
import zipfile
out_zip = Path("/kaggle/working/lab10_results.zip" if ON_KAGGLE else "lab10_results.zip")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list(Path("report").rglob("*")) + list(WORK.glob("*.jsonl")) + list(WORK.glob("*.infer.json")) \\
             + list(WORK.glob("runs/*/run.json")):
        if p.is_file():
            z.write(p, p.as_posix())
print("đã ghi", out_zip, f"{out_zip.stat().st_size / 1e6:.1f} MB")
"""),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                  "isGpuEnabled": True},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = Path(__file__).resolve().parents[1] / "lab_day10.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
