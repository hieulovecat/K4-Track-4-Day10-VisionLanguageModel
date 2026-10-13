"""Test phần Python thuần của lab10 (không cần GPU). Chạy: python -m pytest lab10/tests -q"""
import json

import pytest

from lab10.labels import (entry_answer, gt_boxes_1000, objects_to_answer, reward_iou,
                          yolo_to_box)
from scripts.score import evaluate


def _entry(objects, w=640, h=360):
    return {"image_id": "x", "width": w, "height": h, "objects": objects}


def test_yolo_to_box_norm1000_and_clamp():
    assert yolo_to_box("crack", [0.5, 0.5, 0.2, 0.4]) == {"label": "crack", "bbox_2d": [400, 300, 600, 700]}
    # box tràn mép bị kẹp vào [0, 1000]
    assert yolo_to_box("pothole", [0.02, 0.99, 0.1, 0.1])["bbox_2d"] == [0, 940, 70, 1000]


def test_degenerate_boxes_stay_valid_for_referee():
    """Box mảnh hơn 1 đơn vị, rộng 0, hoặc dính mép phải -> vẫn x2 > x1, y2 > y1 để score.py không loại cả ảnh."""
    from scripts.score import parse_output
    for yolo in ([0.5, 0.5, 0.0004, 0.2], [0.5, 0.5, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
        for coord in ("norm1000", "pixel"):
            e = _entry([{"label": "crack", "yolo": yolo}])
            assert parse_output(entry_answer(e, coord)) is not None, (yolo, coord)
    assert yolo_to_box("crack", [1.0, 1.0, 0.0, 0.0])["bbox_2d"] == [999, 999, 1000, 1000]


def test_yolo_to_box_pixel():
    assert yolo_to_box("manhole", [0.5, 0.5, 0.25, 0.5], "pixel", 640, 360)["bbox_2d"] == [240, 90, 400, 270]


def test_answer_sorted_compact_and_empty():
    objs = [{"label": "crack", "yolo": [0.5, 0.8, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.7, 0.2, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    assert " " not in ans
    assert [b["label"] for b in json.loads(ans)] == ["manhole", "pothole", "crack"]
    assert objects_to_answer([]) == "[]"


def test_answer_scores_perfectly_with_referee():
    objs = [{"label": "crack", "yolo": [0.31, 0.62, 0.05, 0.03]},
            {"label": "pothole", "yolo": [0.7, 0.4, 0.2, 0.1]}]
    e = _entry(objs)
    for coord in ("norm1000", "pixel"):
        res = evaluate([e], {"x": entry_answer(e, coord)}, coord=coord, n_boot=10)
        assert res["micro"]["f1"] == 1.0 and res["mean_iou"] > 0.9, coord


def test_wrong_scale_breaks_f1():
    """Câu 3: đáp án thang 0-1000 nhưng chấm như pixel (và ngược lại) -> F1 sụp."""
    e = _entry([{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}])
    assert evaluate([e], {"x": entry_answer(e, "norm1000")}, coord="pixel", n_boot=10)["micro"]["f1"] == 0.0
    assert evaluate([e], {"x": entry_answer(e, "pixel")}, coord="norm1000", n_boot=10)["micro"]["f1"] == 0.0


GT = gt_boxes_1000(_entry([{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]))
PERFECT = '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'


def test_reward_perfect():
    assert reward_iou(PERFECT, GT) == pytest.approx(1.0)


def test_reward_broken_json():
    assert reward_iou('[{"label":"pothole","bbox_2d":[400,400', GT) == 0.0
    assert reward_iou("there is a pothole", GT) == 0.0


def test_reward_wrong_class_only_format_bonus():
    assert reward_iou('[{"label":"crack","bbox_2d":[400,400,600,600]}]', GT) == pytest.approx(0.2)


def test_reward_extra_box_penalised():
    extra = '[{"label":"pothole","bbox_2d":[400,400,600,600]},{"label":"pothole","bbox_2d":[0,0,50,50]}]'
    assert reward_iou(extra, GT) == pytest.approx(0.2 + 0.8 * 1.0 / 2)


def test_reward_empty_cases():
    assert reward_iou("[]", GT) == pytest.approx(0.2)        # bỏ sót: chỉ còn điểm định dạng
    assert reward_iou("[]", []) == 1.0                        # ảnh sạch, trả rỗng: đúng hoàn toàn
    assert reward_iou(PERFECT, []) == pytest.approx(0.2)      # ảnh sạch mà bịa box


def test_reward_giant_box_not_hacked():
    giant = '[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]'
    assert reward_iou(giant, GT) == pytest.approx(0.2)        # IoU = 0.04 < 0.5


def test_reward_low_iou_ignored_and_partial_iou():
    shifted = '[{"label":"pothole","bbox_2d":[420,400,620,600]}]'  # giao 180x200, hợp 44000
    assert reward_iou(shifted, GT) == pytest.approx(0.2 + 0.8 * 36000 / 44000)
    far = '[{"label":"pothole","bbox_2d":[550,400,750,600]}]'      # IoU = 1/7 < 0.5
    assert reward_iou(far, GT) == pytest.approx(0.2)
