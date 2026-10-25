"""TOMO VLA adapter: Isaac RGB-D/q/instruction -> 30x46 + suction + done."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

# Reuse the verified ACT preprocessing contract.
BASE_ADAPTER = THIS.parent / "act_rgbd_dinov2_adapter.py"
if BASE_ADAPTER.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_act_adapter", BASE_ADAPTER)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    build_obs = base.build_obs
    Obs = base.Obs
    JOINT_NAMES = base.JOINT_NAMES
    N_OBS = base.N_OBS
    HORIZON = base.HORIZON
else:
    raise FileNotFoundError(
        "Place this overlay inside share_act_dinov2 so act_rgbd_dinov2_adapter.py is beside it."
    )

from vla.checkpoint import load_vla_checkpoint
from vla.text_encoder import LanguageBank, SigLIPInstructionEncoder

DEFAULT_BANK = ROOT / "model/siglip_language_bank.pt"


class VLAPolicy:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        text_model: str = "google/siglip-base-patch16-224",
        text_device: str = "cpu",
        local_files_only: bool = True,
        language_bank: str | None = None,
    ):
        self.device = torch.device(device)
        self.model, self.ck = load_vla_checkpoint(checkpoint, self.device, freeze_dino=True)
        self.model.eval()
        self.q_mean = self.ck["q_mean"].float().to(self.device)
        self.q_std = self.ck["q_std"].float().to(self.device)
        # joints that never move in the data (std at the 1e-4 floor): feed the training constant, like
        # the ACT adapter's CLAMP_CONSTANT_JOINTS, so tiny sim drift cannot become a 1e4-sigma input
        self.const = torch.nonzero(self.ck["q_std"].float() <= 1.0001e-4).flatten().to(self.device)
        bank = Path(language_bank) if language_bank else DEFAULT_BANK
        self.bank = LanguageBank(bank) if bank.exists() else None
        self.text_model, self.text_device, self.local_only = text_model, text_device, local_files_only
        self.encoder = None
        self._lang_cache = {}

    def encode_instruction(self, instruction: str):
        """Training bank first (exact same vectors as training); SigLIP only for unseen text."""
        instruction = str(instruction or "")
        if instruction not in self._lang_cache:
            if instruction == "":
                e = self.bank.empty() if self.bank is not None else None
                lang = {"tokens": e["tokens"][None], "mask": e["mask"][None], "sentence": e["sentence"][None]}
            elif self.bank is not None and instruction in self.bank:
                e = self.bank.get(instruction)
                lang = {"tokens": e["tokens"][None], "mask": e["mask"][None], "sentence": e["sentence"][None]}
            else:
                if self.encoder is None:
                    self.encoder = SigLIPInstructionEncoder(self.text_model, device=self.text_device,
                                                            max_tokens=self.model.cfg.max_text_tokens,
                                                            local_files_only=self.local_only)
                lang = self.encoder.encode([instruction])
            self._lang_cache[instruction] = lang
        return self._lang_cache[instruction]

    @torch.inference_mode()
    def predict(self, obs_hist, qs, instruction: str, seed: int = 0):
        torch.manual_seed(int(seed))
        o = obs_hist[-1]
        qraw = np.asarray(qs[-1], np.float32)
        q = torch.from_numpy(qraw.copy()).to(self.device)
        q[self.const] = self.q_mean[self.const]
        qn = ((q - self.q_mean) / self.q_std).unsqueeze(0)
        lang = self.encode_instruction(instruction)
        out = self.model(
            o.rgb.unsqueeze(0).to(self.device),
            o.depth.unsqueeze(0).to(self.device),
            qn,
            lang["tokens"].to(self.device),
            lang["mask"].to(self.device),
            lang["sentence"].to(self.device),
        )
        pred = out["pred"][0] * self.q_std + self.q_mean
        return {
            "horizon": pred.float().cpu().numpy(),
            "suction": torch.sigmoid(out["suction_logits"][0]).float().cpu().numpy(),
            "done": torch.sigmoid(out["done_logits"][0]).float().cpu().numpy(),
            "phone_pose": out["phone_pose"][0].float().cpu().numpy(),
            "phone_patch": int(out["phone_heatmap_logits"][0].argmax().item()),
        }


def load_policy(
    device: str,
    checkpoint: str,
    text_model: str,
    text_device: str = "cpu",
    local_files_only: bool = True,
    language_bank: str | None = None,
):
    return VLAPolicy(checkpoint, device, text_model, text_device, local_files_only, language_bank)
