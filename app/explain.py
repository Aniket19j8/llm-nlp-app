from __future__ import annotations
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Lightweight gradient-based attribution for demo purposes.
# For production-grade explanations, consider Captum's IntegratedGradients.

class TokenAttributor:
    def __init__(self, model_id: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    @torch.inference_mode(False)
    def attribute(self, text: str, steps: int = 16) -> List[Tuple[str, float]]:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Baseline: zeros (very rough; OK for demo)
        baseline_ids = torch.zeros_like(input_ids)

        accumulated_grads = torch.zeros_like(input_ids, dtype=torch.float32)
        self.model.zero_grad(set_to_none=True)

        for alpha in torch.linspace(0, 1, steps):
            interp_ids = (baseline_ids * (1 - alpha) + input_ids * alpha).round().long()
            interp_embeds = self.model.get_input_embeddings()(interp_ids)
            interp_embeds.requires_grad_(True)

            outputs = self.model(inputs_embeds=interp_embeds, attention_mask=attention_mask)
            # Target the predicted class
            pred = outputs.logits.softmax(-1).max(dim=-1).values.sum()
            pred.backward(retain_graph=True)

            grads = interp_embeds.grad.detach().abs().sum(dim=-1)
            accumulated_grads += grads
            self.model.zero_grad(set_to_none=True)

        scores = accumulated_grads[0]
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        eps = 1e-9
        smax = scores.max().item() + eps
        pairs = [(tok, (scores[i].item() / smax)) for i, tok in enumerate(token_list)]
        pairs = [(t, s) for (t, s) in pairs if t not in ("[CLS]", "[SEP]", "[PAD]")]
        return pairs
