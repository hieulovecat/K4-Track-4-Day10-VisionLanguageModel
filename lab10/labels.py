"""Phần Python thuần (không cần GPU): nhãn YOLO -> JSON, prompt, reward IoU cho GRPO."""
import json

from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output

PROMPT_NORM1000 = (
    "Detect all road damage in this dashcam image. "
    "Classes: pothole, crack, manhole. "
    'Return ONLY a JSON list like [{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}] '
    "with one item per damage instance. Coordinates are integers on a 0-1000 scale "
    "relative to the image width and height: (x1, y1) is the top-left corner and "
    "(x2, y2) the bottom-right corner. If there is no damage, return []."
)


def prompt_pixel(width, height):
    return (
        "Detect all road damage in this dashcam image. "
        "Classes: pothole, crack, manhole. "
        'Return ONLY a JSON list like [{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}] '
        "with one item per damage instance. Coordinates are integer pixel positions in the "
        f"{width}x{height} image: (x1, y1) is the top-left corner and (x2, y2) the "
        "bottom-right corner. If there is no damage, return []."
    )


def get_prompt(entry, coord="norm1000"):
    return PROMPT_NORM1000 if coord == "norm1000" else prompt_pixel(entry["width"], entry["height"])


def _clamp(v, hi):
    return max(0, min(hi, int(round(v))))


def yolo_to_box(label, yolo, coord="norm1000", width=None, height=None):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) -> {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    return {"label": label, "bbox_2d": [
        _clamp((xc - w / 2) * sx, sx), _clamp((yc - h / 2) * sy, sy),
        _clamp((xc + w / 2) * sx, sx), _clamp((yc + h / 2) * sy, sy)]}


def objects_to_answer(objects, coord="norm1000", width=None, height=None):
    """Chuỗi JSON đáp án, box sắp theo (y1, x1): trên->dưới, trái->phải."""
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def entry_answer(entry, coord="norm1000"):
    return objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])


def gt_boxes_1000(entry):
    """Box đúng ở thang 0-1000 (float, giống trọng tài score.py)."""
    out = []
    for o in entry["objects"]:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def box_1000_to_pixel(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)


def reward_iou(completion, gt, iou_thr=0.5):
    """Reward kiểm chứng được cho GRPO.

    gt: list[{"label", "bbox"}] ở thang 0-1000 (xem gt_boxes_1000).
    0.0 nếu JSON hỏng; 1.0 nếu cả gt lẫn dự đoán đều rỗng; còn lại
    0.2 + 0.8 * tổng IoU các cặp khớp (cùng lớp, IoU >= 0.5, ghép tham lam) / max(n_gt, n_pred, 1).
    Box bịa (thừa) làm tăng mẫu số nên bị phạt.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)


__all__ = ["CLASSES", "PROMPT_NORM1000", "prompt_pixel", "get_prompt", "yolo_to_box",
           "objects_to_answer", "entry_answer", "gt_boxes_1000", "box_1000_to_pixel",
           "reward_iou", "parse_output"]
