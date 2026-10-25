from __future__ import annotations
from pathlib import Path
from typing import Iterable
import torch


DEFAULT_MODEL = "google/siglip-base-patch16-224"
SIGLIP_CONTEXT = 64          # SigLIP text was trained with padding="max_length", 64 tokens


class SigLIPInstructionEncoder:
    """Frozen SigLIP text tower (same code at training time and at deployment).

    * The text is tokenized exactly like SigLIP training: padding="max_length" to 64 tokens and
      NO attention mask passed to the model, so the pooled sentence vector is the one SigLIP learned.
    * The word-token path of the VLA uses the first `max_tokens` (16) hidden states; its mask marks
      the real (non-padding) tokens. The mask is computed from the un-padded token count, so it does
      not depend on whether this tokenizer returns an attention_mask (SiglipTokenizer does not).
    Training uses the precomputed LanguageBank; deployment encodes once per new instruction (CPU).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_tokens: int = 16,
        dtype: str = "float32",
        local_files_only: bool = False,
    ):
        from transformers import AutoTokenizer, SiglipTextModel

        self.model_id = model_id
        self.max_tokens = int(max_tokens)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = SiglipTextModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if dtype == "float16" and self.device.type == "cuda":
            self.model.half()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        texts = [str(t) for t in texts]
        ctx = min(SIGLIP_CONTEXT, int(getattr(self.model.config, "max_position_embeddings", SIGLIP_CONTEXT)))
        tok = self.tokenizer(texts, padding="max_length", truncation=True, max_length=ctx,
                             return_tensors="pt")
        # real token count per text (incl. the end token), independent of attention_mask support
        lengths = [min(len(ids), ctx) for ids in self.tokenizer(texts, truncation=True, max_length=ctx)["input_ids"]]
        out = self.model(input_ids=tok["input_ids"].to(self.device))
        hidden = out.last_hidden_state.float().cpu()                     # [B, ctx, D]
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:                                               # fallback: masked mean
            m = (torch.arange(ctx)[None] < torch.tensor(lengths)[:, None]).float().unsqueeze(-1)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
        pooled = pooled.float().cpu()
        k = self.max_tokens
        mask = torch.arange(k)[None] < torch.tensor(lengths)[:, None]    # [B, k]
        if not bool(mask.any(dim=1).all()):
            raise ValueError("an instruction produced no tokens")
        return {"tokens": hidden[:, :k].contiguous(), "mask": mask, "sentence": pooled}


class LanguageBank:
    """Exact-text -> precomputed SigLIP tokens [16,768], mask [16] and sentence vector [768]."""

    def __init__(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model_id = payload["model_id"]
        self.max_tokens = int(payload["max_tokens"])
        self.text_dim = int(payload["text_dim"])
        self.items = payload["items"]

    def __contains__(self, text: str) -> bool:
        return text in self.items

    def get(self, text: str) -> dict[str, torch.Tensor]:
        if text not in self.items:
            raise KeyError(f"Instruction not in language bank: {text!r}. Run the 'language' step again.")
        x = self.items[text]
        return {"tokens": x["tokens"].float(), "mask": x["mask"].bool(), "sentence": x["sentence"].float()}

    def empty(self) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.zeros(self.max_tokens, self.text_dim),
            "mask": torch.zeros(self.max_tokens, dtype=torch.bool),
            "sentence": torch.zeros(self.text_dim),
        }


def save_language_bank(path: str | Path, texts: list[str], encoder: SigLIPInstructionEncoder,
                       batch_size: int = 32):
    items = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = encoder.encode(batch)
        for j, text in enumerate(batch):
            items[text] = {"tokens": enc["tokens"][j].half(), "mask": enc["mask"][j],
                           "sentence": enc["sentence"][j].half()}
    payload = {"model_id": encoder.model_id, "max_tokens": encoder.max_tokens,
               "text_dim": encoder.hidden_size, "items": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
