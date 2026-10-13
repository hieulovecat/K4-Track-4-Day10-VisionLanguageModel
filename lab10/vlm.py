"""Phần GPU: nạp Qwen3-VL-4B 4-bit, mã hoá ví dụ + mask loss, LoRA/đóng băng, huấn luyện, suy luận, GRPO.

Mọi hàm nhận model/processor làm tham số để có thể chạy thử trên CPU với một model tí hon.
"""
import gc
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from lab10.labels import entry_answer, get_prompt, gt_boxes_1000, reward_iou
from scripts.labcommon import read_jsonl
from scripts.score import parse_output

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Tên module lấy từ cây model thật (đã kiểm tra: 252 module LLM, 96 module ViT).
LLM_TARGETS = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc(?:1|2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")

# transformers 5.x so khớp module bằng re.match (neo ở đầu tên) hoặc endswith, nên "visual" một mình
# KHÔNG khớp "model.visual.blocks.0..." -> ViT bị lượng tử hoá. Thêm "model.visual" để chắc chắn.
SKIP_QUANT = ["visual", "model.visual", "lm_head"]


# ----------------------------------------------------------------------------- nạp model

def load_base(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def quantization_report(model):
    """Đếm Linear 4-bit trong ViT và LLM. ViT phải = 0 (để huấn luyện được projector ở E2/E3)."""
    counts = {"visual_4bit": 0, "llm_4bit": 0, "visual_linear_fp": 0}
    for n, m in model.named_modules():
        is4 = type(m).__name__ == "Linear4bit"
        if "visual." in n:
            if is4:
                counts["visual_4bit"] += 1
            elif isinstance(m, torch.nn.Linear):
                counts["visual_linear_fp"] += 1
        elif is4:
            counts["llm_4bit"] += 1
    return counts


def param_breakdown(model_id=MODEL_ID):
    """Số tham số từng khối, tính trên meta device (không tải trọng số, không bị nén 4-bit làm sai)."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        m = Qwen3VLForConditionalGeneration(cfg)

    def cnt(pred):
        return sum(p.numel() for n, p in m.named_parameters() if pred(n))

    out = {
        "vit_total": cnt(lambda n: ".visual." in f".{n}"),
        "vit_blocks+patch_embed": cnt(lambda n: ".visual." in f".{n}" and not any(k in n for k in PROJECTOR_KEYS)),
        "projector_merger": cnt(lambda n: "visual.merger" in n),
        "projector_deepstack": cnt(lambda n: "visual.deepstack_merger_list" in n),
        "llm_total(incl. embedding)": cnt(lambda n: "language_model" in n),
        "llm_embed_tokens": cnt(lambda n: "embed_tokens" in n),
        "lm_head(untied)": cnt(lambda n: n.startswith("lm_head")),
        "total": cnt(lambda n: True),
    }
    names = [n for n, _ in m.named_modules()]
    out["lora_modules_llm"] = sum(bool(re.fullmatch(LLM_TARGETS, n)) for n in names)
    out["lora_modules_vit"] = sum(bool(re.fullmatch(VIT_TARGETS, n)) for n in names)
    del m
    return out


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device(model):
    return next(model.parameters()).device


def _amp(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


# ----------------------------------------------------------------------------- dữ liệu

def load_image(entry, data_root="data"):
    with Image.open(Path(data_root) / entry["image_path"]) as im:
        return im.convert("RGB")


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    """Chuỗi đầy đủ (prompt + đáp án) với labels = -100 trên toàn bộ prompt (kể cả token ảnh)."""
    user = user_msg(image, prompt)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n], p["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")
    out = dict(full)
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    out["labels"] = labels
    return out


def collate(features, pad_id):
    """Pad phải các tensor theo chuỗi; pixel_values / image_grid_thw nối theo chiều 0, không pad."""
    L = max(f["input_ids"].shape[1] for f in features)
    out = {}
    for k in features[0]:
        if k in ("pixel_values", "image_grid_thw"):
            out[k] = torch.cat([f[k] for f in features], 0)
        else:
            val = {"input_ids": pad_id, "labels": -100}.get(k, 0)
            out[k] = torch.cat([F.pad(f[k], (0, L - f[k].shape[1]), value=val) for f in features], 0)
    return out


def image_token_count(processor, enc):
    pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return int((enc["input_ids"] == pad).sum())


# ----------------------------------------------------------------------------- LoRA / đóng băng

def apply_setup(model, setup, r=16, alpha=32, dropout=0.05):
    """E1: LoRA LLM. E2: + projector đầy đủ (fp32). E3: + LoRA ViT. Trả (peft_model, info)."""
    from peft import LoraConfig, get_peft_model

    setup = setup.upper()
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    pattern = LLM_TARGETS if setup in ("E1", "E2") else f"(?:{LLM_TARGETS})|(?:{VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=pattern, task_type="CAUSAL_LM"))
    n_proj = 0
    if setup in ("E2", "E3"):
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
                n_proj += p.numel()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, p in trainable if p.dtype != torch.float32]
    if bad:  # GradScaler không unscale được gradient fp16
        raise TypeError(f"tham số huấn luyện phải là float32: {bad[:3]}")
    info = {"setup": setup, "lora_r": r, "lora_alpha": alpha, "matched_modules": len(matched),
            "projector_params": n_proj, "trainable_params": sum(p.numel() for _, p in trainable)}
    print(json.dumps(info))
    return model, info


def projector_state(model):
    return {n[n.index("visual."):]: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}


def save_trainable(model, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")        # chỉ chứa LoRA
    proj = projector_state(model)                     # projector KHÔNG nằm trong adapter
    if proj:
        torch.save(proj, out_dir / "projector.pt")


def attach_trained(model, run_dir, is_trainable=False):
    """Nạp adapter LoRA + projector đã huấn luyện (nếu có) lên model gốc."""
    from peft import PeftModel

    run_dir = Path(run_dir)
    model = PeftModel.from_pretrained(model, str(run_dir / "adapter"), is_trainable=is_trainable)
    proj_path = run_dir / "projector.pt"
    if proj_path.exists():
        sd = torch.load(proj_path, map_location="cpu")
        params = {n[n.index("visual."):]: p for n, p in model.named_parameters() if "visual." in n}
        missing = sorted(set(sd) - set(params))
        if missing:
            raise KeyError(f"không tìm thấy tham số projector: {missing[:3]}")
        with torch.no_grad():
            for k, v in sd.items():
                params[k].copy_(v.to(params[k].dtype))
        print(f"đã nạp lại {len(sd)} tensor projector ({sum(v.numel() for v in sd.values()) / 1e6:.1f}M tham số)")
    model.eval()
    return model


# ----------------------------------------------------------------------------- huấn luyện

def train(model, processor, entries, out_dir, *, coord="norm1000", lr=2e-4, batch_size=2,
          grad_accum=4, max_steps=300, seed=42, save_every=100, log_every=10, data_root="data",
          extra_info=None):
    from transformers import get_cosine_schedule_with_warmup

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(model)
    pad_id = processor.tokenizer.pad_token_id
    rng = random.Random(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def batches():
        while True:
            order = list(entries)
            rng.shuffle(order)
            for i in range(0, len(order) - batch_size + 1, batch_size):
                yield order[i:i + batch_size]

    it = batches()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, log, window = time.time(), [], []
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            feats = [encode_example(processor, load_image(e, data_root), get_prompt(e, coord),
                                    entry_answer(e, coord)) for e in next(it)]
            batch = {k: v.to(device) for k, v in collate(feats, pad_id).items()}
            with _amp(device):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        window.append(step_loss)
        if step % log_every == 0 or step == max_steps:
            rec = {"step": step, "loss": sum(window) / len(window), "grad_norm": gnorm,
                   "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
            log.append(rec)
            window = []
            print(f"step {step:4d}  loss {rec['loss']:.4f}  gnorm {gnorm:.2f}  "
                  f"lr {rec['lr']:.2e}  {rec['minutes']:.1f} phút", flush=True)
        if step % save_every == 0 or step == max_steps:
            save_trainable(model, out_dir)
    run = {
        **(extra_info or {}),
        "coord": coord, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
        "max_steps": max_steps, "seed": seed, "n_train_images": len(entries),
        "trainable_params": sum(p.numel() for p in params),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0,
        "train_minutes": (time.time() - t0) / 60,
        "loss_log": log, "done": True,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


# ----------------------------------------------------------------------------- suy luận + chấm

@torch.no_grad()
def predict(model, processor, entries, out_path, *, coord="norm1000", batch_size=8,
            max_new_tokens=256, data_root="data", log_every=10):
    """Ghi {"image_id","raw"} mỗi dòng; chạy tiếp được nếu bị ngắt giữa chừng."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    device = _device(model)
    processor.tokenizer.padding_side = "left"
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for bi, i in enumerate(range(0, len(todo), batch_size)):
            batch = todo[i:i + batch_size]
            msgs = [[user_msg(load_image(e, data_root), get_prompt(e, coord))] for e in batch]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(device)
            with _amp(device):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, t in zip(batch, texts):
                f.write(json.dumps({"image_id": e["image_id"], "raw": t}, ensure_ascii=False) + "\n")
            f.flush()
            if (bi + 1) % log_every == 0:
                print(f"  {i + len(batch)}/{len(todo)} ảnh, {(time.time() - t0) / 60:.1f} phút", flush=True)
    return {"infer_minutes": (time.time() - t0) / 60,
            "infer_peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0,
            "n_new": len(todo)}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"  # luôn có dấu "." để score.py đọc thành số thực
    return str(v)


def score(pred_path, name, out_path, *, split="test", coord="norm1000", meta=None,
          splits_dir="data/splits"):
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred_path), "--name", name,
           "--out", str(out_path), "--split", split, "--coord", coord, "--splits-dir", str(splits_dir)]
    for k, v in (meta or {}).items():
        cmd += ["--meta", f"{k}={_fmt(v)}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    r.check_returncode()
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- GRPO (E5)

def grpo_train(model, processor, entries, out_dir, *, steps=60, num_generations=4,
               max_completion_length=128, lr=5e-6, prompts_per_step=2, temperature=1.0,
               seed=42, data_root="data", init_dir=None):
    """GRPO on-policy, beta = 0 (không KL), một cập nhật cho mỗi lô sinh.

    Với mỗi ảnh: sinh G câu trả lời, reward = reward_iou, advantage = (r - mean) / std trong nhóm,
    loss = -mean_i(adv_i * logp_trung_bình_token_i). Khi chỉ cập nhật một lần trên dữ liệu vừa sinh,
    tỉ số pi/pi_old = 1 nên đây đúng là gradient của GRPO (clip không kích hoạt).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(model)
    tok = processor.tokenizer
    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=device)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        rewards_all, n_empty, n_invalid, areas, lens, n_updates = [], 0, 0, [], [], 0
        for _ in range(prompts_per_step):
            e = rng.choice(entries)
            enc = processor.apply_chat_template(
                [[user_msg(load_image(e, data_root), get_prompt(e))]], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
            n = enc["input_ids"].shape[1]
            model.eval()
            with torch.no_grad(), _amp(device):
                seq = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                                     max_new_tokens=max_completion_length,
                                     num_return_sequences=num_generations,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            comp = seq[:, n:]
            texts = tok.batch_decode(comp, skip_special_tokens=True)
            gt = gt_boxes_1000(e)
            rewards = torch.tensor([reward_iou(t, gt) for t in texts], dtype=torch.float32)
            rewards_all += rewards.tolist()
            for t in texts:
                s = t.strip()
                if s == "[]":
                    n_empty += 1
                p = parse_output(t)
                if p is None:
                    n_invalid += 1
                else:
                    areas += [(b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) / 1e6 for b in p]
            # mặt nạ: token đến hết eos đầu tiên (kể cả eos)
            is_eos = torch.isin(comp, eos)
            first = torch.where(is_eos.any(1), is_eos.float().argmax(1),
                                torch.full((comp.shape[0],), comp.shape[1] - 1, device=device))
            mask = (torch.arange(comp.shape[1], device=device)[None] <= first[:, None]).float()
            lens += mask.sum(1).tolist()
            std = rewards.std()
            if std < 1e-6:          # cả nhóm cùng reward: advantage = 0, không có tín hiệu
                continue
            adv = ((rewards - rewards.mean()) / (std + 1e-4)).to(device)
            G = comp.shape[0]
            batch = {"input_ids": torch.cat([enc["input_ids"].repeat(G, 1), comp], 1),
                     "attention_mask": torch.cat([enc["attention_mask"].repeat(G, 1), mask.long()], 1),
                     "pixel_values": enc["pixel_values"].repeat(G, 1),
                     "image_grid_thw": enc["image_grid_thw"].repeat(G, 1)}
            for k, v in enc.items():
                if k not in batch and torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n:
                    batch[k] = torch.cat([v.repeat(G, 1), torch.zeros_like(comp)], 1)
            model.train()
            with _amp(device):
                logits = model(**batch, logits_to_keep=comp.shape[1] + 1).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            seq_logp = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = -(adv * seq_logp).mean() / prompts_per_step
            scaler.scale(loss).backward()
            n_updates += 1
        if n_updates:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "reward_mean": sum(rewards_all) / len(rewards_all),
               "reward_max": max(rewards_all), "frac_empty": n_empty / len(rewards_all),
               "frac_invalid": n_invalid / len(rewards_all),
               "mean_box_area": sum(areas) / len(areas) if areas else 0.0,
               "mean_len": sum(lens) / len(lens), "updates": n_updates,
               "minutes": (time.time() - t0) / 60}
        log.append(rec)
        print(f"step {step:3d} reward {rec['reward_mean']:.3f} empty {rec['frac_empty']:.2f} "
              f"invalid {rec['frac_invalid']:.2f} area {rec['mean_box_area']:.4f} "
              f"len {rec['mean_len']:.0f} upd {n_updates} {rec['minutes']:.1f} phút", flush=True)
    model.save_pretrained(out_dir / "adapter")
    if init_dir and (Path(init_dir) / "projector.pt").exists():  # projector giữ nguyên từ SFT
        shutil.copy2(Path(init_dir) / "projector.pt", out_dir / "projector.pt")
    run = {"steps": steps, "num_generations": num_generations, "max_completion_length": max_completion_length,
           "lr": lr, "prompts_per_step": prompts_per_step, "temperature": temperature, "beta": 0.0,
           "init": str(init_dir), "trainable_params": sum(p.numel() for p in params),
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0,
           "train_minutes": (time.time() - t0) / 60, "log": log, "done": True}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def max_answer_tokens(processor, entries, coord="norm1000"):
    return max(len(processor.tokenizer(entry_answer(e, coord))["input_ids"]) for e in entries)

