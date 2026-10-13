"""Điều phối thí nghiệm: huấn luyện -> suy luận -> chấm, chạy tiếp được nếu phiên bị ngắt."""
import json
import random
from pathlib import Path

from lab10 import vlm
from scripts.labcommon import read_jsonl

# Siêu tham số chung, khớp configs/e1.yaml..e3.yaml
BASE = {"lr": 2e-4, "batch_size": 2, "grad_accum": 4, "max_steps": 300, "seed": 42,
        "r": 16, "alpha": 32, "coord": "norm1000", "train_fraction": 1.0}
CONFIGS = {
    "e1": {**BASE, "key": "e1", "name": "E1", "setup": "E1"},
    "e2": {**BASE, "key": "e2", "name": "E2", "setup": "E2"},
    "e3": {**BASE, "key": "e3", "name": "E3", "setup": "E3"},
}


class Lab:
    def __init__(self, splits, *, eval_entries, split="test", evid_dir="report/evidence",
                 work_dir="work", loader=vlm.load_base, infer_bs=8, max_new_tokens=256,
                 data_root="data", overrides=None):
        self.splits = splits
        self.eval_entries = eval_entries
        self.split = split
        self.evid = Path(evid_dir)
        self.work = Path(work_dir)
        self.loader = loader
        self.infer_bs = infer_bs
        self.max_new_tokens = max_new_tokens
        self.data_root = data_root
        self.overrides = overrides or {}   # vd {"max_steps": 4} khi chạy thử
        self.evid.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- tiện ích
    def cfg(self, cfg):
        return {**cfg, **self.overrides}

    def run_dir(self, key):
        return self.work / "runs" / key

    def pred_path(self, key):
        return self.work / f"{key}.jsonl"

    def train_entries(self, cfg):
        train = self.splits["train"]
        frac = cfg.get("train_fraction", 1.0)
        if frac >= 1.0:
            return train
        return random.Random(cfg["seed"]).sample(train, max(1, round(frac * len(train))))

    def _preds_complete(self, key):
        p = self.pred_path(key)
        if not p.exists():
            return False
        done = {r["image_id"] for r in read_jsonl(p)}
        return all(e["image_id"] in done for e in self.eval_entries)

    def _predict(self, model, processor, key, coord):
        side = self.work / f"{key}.infer.json"
        st = vlm.predict(model, processor, self.eval_entries, self.pred_path(key), coord=coord,
                         batch_size=self.infer_bs, max_new_tokens=self.max_new_tokens,
                         data_root=self.data_root)
        prev = json.loads(side.read_text()) if side.exists() else {"infer_minutes": 0.0, "infer_peak_vram_gb": 0.0}
        if st["n_new"]:
            prev = {"infer_minutes": prev["infer_minutes"] + st["infer_minutes"],
                    "infer_peak_vram_gb": max(prev["infer_peak_vram_gb"], st["infer_peak_vram_gb"])}
            side.write_text(json.dumps(prev), encoding="utf-8")
        return prev

    def _infer_stats(self, key):
        side = self.work / f"{key}.infer.json"
        return json.loads(side.read_text()) if side.exists() else {}

    def evidence(self, key):
        return json.loads((self.evid / f"{key}.json").read_text(encoding="utf-8"))

    # --------------------------------------------------------------- E0
    def zero_shot(self, model=None, processor=None, key="e0", name="E0"):
        if not self._preds_complete(key):
            own = model is None
            if own:
                model, processor = self.loader()
            self._predict(model, processor, key, "norm1000")
            if own:
                del model
                vlm.free_memory()
        meta = {"trainable_params": 0, **self._infer_stats(key)}
        return vlm.score(self.pred_path(key), name, self.evid / f"{key}.json", split=self.split, meta=meta)

    # --------------------------------------------------------------- huấn luyện
    def train(self, cfg):
        cfg = self.cfg(cfg)
        rd = self.run_dir(cfg["key"])
        rj = rd / "run.json"
        if rj.exists() and json.loads(rj.read_text()).get("done"):
            print(f"{cfg['key']}: đã huấn luyện xong, bỏ qua ({rj})")
            return json.loads(rj.read_text())
        model, processor = self.loader()
        model, info = vlm.apply_setup(model, cfg["setup"], r=cfg["r"], alpha=cfg["alpha"])
        entries = self.train_entries(cfg)
        run = vlm.train(model, processor, entries, rd, coord=cfg["coord"], lr=cfg["lr"],
                        batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
                        max_steps=cfg["max_steps"], seed=cfg["seed"], data_root=self.data_root,
                        extra_info={**info, "config": cfg})
        del model
        vlm.free_memory()
        print(f"{cfg['key']}: trainable={run['trainable_params']:,}  VRAM đỉnh={run['peak_vram_gb']:.2f} GB  "
              f"{run['train_minutes']:.1f} phút")
        return run

    # --------------------------------------------------------------- suy luận + chấm
    def evaluate(self, cfg, evidence_key=None, name=None):
        cfg = self.cfg(cfg)
        key = cfg["key"]
        run = json.loads((self.run_dir(key) / "run.json").read_text())
        if not self._preds_complete(key):
            model, processor = self.loader()
            model = vlm.attach_trained(model, self.run_dir(key))
            self._predict(model, processor, key, cfg["coord"])
            del model
            vlm.free_memory()
        meta = {"trainable_params": run["trainable_params"], "peak_vram_gb": run["peak_vram_gb"],
                "train_minutes": run["train_minutes"], **self._infer_stats(key)}
        ek = evidence_key or key
        return vlm.score(self.pred_path(key), name or cfg["name"], self.evid / f"{ek}.json",
                         split=self.split, coord=cfg["coord"], meta=meta)

    def run(self, cfg, evidence_key=None, name=None):
        self.train(cfg)
        return self.evaluate(cfg, evidence_key, name)

    def rescore(self, key, evidence_path, name, coord="norm1000", extra_meta=None):
        """Chấm lại dự đoán có sẵn (không suy luận lại), vd đổi --coord hoặc tái dùng setup gốc cho E4."""
        meta = dict(self.evidence(key).get("meta", {})) if (self.evid / f"{key}.json").exists() else {}
        meta.update(extra_meta or {})
        return vlm.score(self.pred_path(key), name, evidence_path, split=self.split, coord=coord, meta=meta)
