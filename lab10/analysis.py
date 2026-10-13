"""Phân tích lỗi, chọn ví dụ và vẽ box dự đoán / box đúng (chỉ cần PIL)."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lab10.labels import CLASSES, box_1000_to_pixel, gt_boxes_1000
from scripts.score import evaluate, parse_output

GT_COLOR = (0, 220, 0)
PRED_COLOR = (255, 40, 40)
_FENCE = re.compile(r"```")


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def per_image(entries, preds, coord="norm1000"):
    """TP/FP/FN từng ảnh theo đúng trọng tài score.py."""
    rows = []
    for e in entries:
        raw = preds.get(e["image_id"])
        r = evaluate([e], {e["image_id"]: raw} if raw is not None else {}, coord=coord, n_boot=1)
        m = r["micro"]
        rows.append({"image_id": e["image_id"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                     "n_gt": len(e["objects"]), "valid": r["parse_valid_rate"] == 1.0})
    return rows


def raw_error_stats(entries, preds):
    """Đếm các kiểu lỗi trong output thô (dùng cho E0, câu 2)."""
    st = {"n": 0, "invalid_json": 0, "code_fence": 0, "extra_text": 0, "empty_list": 0,
          "empty_but_has_gt": 0, "coord_gt_1000": 0, "unknown_label": 0, "boxes_pred": 0,
          "boxes_gt": 0}
    labels = {}
    for e in entries:
        raw = preds.get(e["image_id"], "")
        st["n"] += 1
        st["boxes_gt"] += len(e["objects"])
        if _FENCE.search(raw):
            st["code_fence"] += 1
        p = parse_output(raw)
        if p is None:
            st["invalid_json"] += 1
            continue
        s = raw.strip()
        if not (s.startswith("[") and s.endswith("]")) and not _FENCE.search(raw):
            st["extra_text"] += 1
        if not p:
            st["empty_list"] += 1
            if e["objects"]:
                st["empty_but_has_gt"] += 1
        st["boxes_pred"] += len(p)
        for b in p:
            labels[b["label"]] = labels.get(b["label"], 0) + 1
            if b["label"] not in CLASSES:
                st["unknown_label"] += 1
            if max(b["bbox"]) > 1000:
                st["coord_gt_1000"] += 1
    st["pred_labels"] = dict(sorted(labels.items(), key=lambda kv: -kv[1]))
    return st


def draw_example(entry, raw, out_path, *, coord="norm1000", data_root="data", title=""):
    """Box đúng màu xanh lá, box dự đoán màu đỏ; dòng chú thích bên dưới."""
    with Image.open(Path(data_root) / entry["image_path"]) as im:
        im = im.convert("RGB")
    W, H = im.size
    canvas = Image.new("RGB", (W, H + 46), (255, 255, 255))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(13)
    for g in gt_boxes_1000(entry):
        b = box_1000_to_pixel(g["bbox"], W, H)
        d.rectangle(b, outline=GT_COLOR, width=3)
        d.text((b[0] + 2, max(0, b[1] - 15)), g["label"], fill=GT_COLOR, font=f)
    pred = parse_output(raw or "")
    for p in pred or []:
        b = p["bbox"] if coord == "pixel" else box_1000_to_pixel(p["bbox"], W, H)
        d.rectangle(b, outline=PRED_COLOR, width=2)
        d.text((b[0] + 2, b[3] + 1), p["label"], fill=PRED_COLOR, font=f)
    r = evaluate([entry], {entry["image_id"]: raw or ""}, coord=coord, n_boot=1)["micro"]
    # chú thích ASCII: font có sẵn trên Kaggle/Colab có thể thiếu dấu tiếng Việt
    cap = (f"{title}  {entry['image_id']}\ngreen = ground truth ({len(entry['objects'])}), "
           f"red = prediction ({'INVALID JSON' if pred is None else len(pred)})  "
           f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")
    d.multiline_text((6, H + 4), cap, fill=(0, 0, 0), font=f, spacing=4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return out_path


def pick_examples(entries, preds, coord="norm1000", n_good=3, n_bad=3):
    """Ảnh đúng nhiều nhất và sai nhiều nhất (ưu tiên đủ cả 3 lớp)."""
    rows = {r["image_id"]: r for r in per_image(entries, preds, coord)}
    by_id = {e["image_id"]: e for e in entries}

    def classes(i):
        return {o["label"] for o in by_id[i]["objects"]}

    good = sorted((r for r in rows.values() if r["tp"] > 0),
                  key=lambda r: (r["fp"] + r["fn"] - r["tp"], -r["n_gt"]))
    bad = sorted(rows.values(), key=lambda r: -(r["fp"] + r["fn"]))

    def diverse(cands, k):
        out, seen = [], set()
        for r in cands:                       # lượt 1: mỗi ảnh mang thêm lớp mới
            if classes(r["image_id"]) - seen and len(out) < k:
                out.append(r["image_id"])
                seen |= classes(r["image_id"])
        for r in cands:                       # lượt 2: lấp đủ k
            if len(out) >= k:
                break
            if r["image_id"] not in out:
                out.append(r["image_id"])
        return out

    g = diverse(good, n_good)
    return g, diverse([r for r in bad if r["image_id"] not in g], n_bad)


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def paired_bootstrap(entries, preds_a, preds_b, *, coord_a="norm1000", coord_b="norm1000",
                     n_boot=2000, seed=0):
    """CI95 của hiệu F1(B) - F1(A), lấy mẫu lại CÙNG tập ảnh cho cả hai (chặt hơn so hai CI rời)."""
    import random

    a = per_image(entries, preds_a, coord_a)
    b = per_image(entries, preds_b, coord_b)
    rng, n, diffs = random.Random(seed), len(entries), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [sum(a[i][k] for i in idx) for k in ("tp", "fp", "fn")]
        sb = [sum(b[i][k] for i in idx) for k in ("tp", "fp", "fn")]
        diffs.append(_f1(*sb) - _f1(*sa))
    diffs.sort()
    full_a = _f1(*(sum(r[k] for r in a) for k in ("tp", "fp", "fn")))
    full_b = _f1(*(sum(r[k] for r in b) for k in ("tp", "fp", "fn")))
    return {"f1_a": full_a, "f1_b": full_b, "diff": full_b - full_a,
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "p_b_better": sum(d > 0 for d in diffs) / n_boot}


def summary_row(ev):
    """Một dòng bảng tổng hợp từ file bằng chứng score.py."""
    pc, meta = ev["per_class"], ev.get("meta", {})
    return {"name": ev["name"], "trainable_params": meta.get("trainable_params", 0),
            "peak_vram_gb": meta.get("peak_vram_gb", meta.get("infer_peak_vram_gb")),
            "train_minutes": meta.get("train_minutes"), "f1": ev["micro"]["f1"],
            "ci95": ev["f1_ci95"], "fp_per_image": ev["fp_per_image"],
            "parse_valid_rate": ev["parse_valid_rate"], "mean_iou": ev["mean_iou"],
            **{f"f1_{c}": pc[c]["f1"] for c in CLASSES}}


def markdown_table(evidence_paths):
    head = ("| Mã | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | parse | "
            "F1 pothole | F1 crack | F1 manhole |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for p in evidence_paths:
        r = summary_row(json.loads(Path(p).read_text(encoding="utf-8")))

        def n(v, f="{:.2f}"):
            return "" if v is None else f.format(v)
        lines.append(
            f"| {r['name']} | {r['trainable_params']:,} | {n(r['peak_vram_gb'])} | {n(r['train_minutes'], '{:.1f}')} | "
            f"{r['f1']:.3f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | {r['fp_per_image']:.2f} | "
            f"{r['parse_valid_rate']:.3f} | {r['f1_pothole']:.3f} | {r['f1_crack']:.3f} | {r['f1_manhole']:.3f} |")
    return "\n".join(lines)
