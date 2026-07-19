####管的是router network本身
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from model_backbone_specs import get_decoder_layers, get_pre_attn_norm, infer_backbone_spec

###外部 BERT，回傳倒數第二層和最後一層 hidden states。
class BertExternalEncoder(nn.Module):
    def __init__(self, bert_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_name_or_path)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return out.hidden_states[-2], out.hidden_states[-1]

#####舊版 router head，直接輸出 task logits。
class CompactCrossAttentionRouter(nn.Module):
    def __init__(self, llama_hidden_size: int, bert_hidden_size: int, router_dim: int, num_tasks: int):
        super().__init__()
        self.q_proj = nn.Linear(llama_hidden_size, router_dim)
        self.k_proj = nn.Linear(bert_hidden_size, router_dim)
        self.v_proj = nn.Linear(bert_hidden_size, router_dim)
        self.out_norm = nn.LayerNorm(router_dim * 2)
        self.classifier = nn.Linear(router_dim * 2, num_tasks)

    def forward(self, llama_vec, bert_prev, bert_last, bert_attention_mask=None):
        router_dtype = self.q_proj.weight.dtype
        router_device = self.q_proj.weight.device

        llama_vec = llama_vec.to(device=router_device, dtype=router_dtype)
        bert_prev = bert_prev.to(device=router_device, dtype=router_dtype)
        bert_last = bert_last.to(device=router_device, dtype=router_dtype)

        q = self.q_proj(llama_vec).unsqueeze(1)
        mem = torch.cat([bert_prev, bert_last], dim=1)
        k = self.k_proj(mem)
        v = self.v_proj(mem)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        if bert_attention_mask is not None:
            mask = torch.cat([bert_attention_mask, bert_attention_mask], dim=1)
            mask = (mask == 0).unsqueeze(1).to(device=router_device)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).squeeze(1)
        qv = q.squeeze(1)

        feat = torch.cat([qv, ctx], dim=-1)
        feat = self.out_norm(feat)
        return self.classifier(feat)

###新版 joint router 用的 feature encoder，不直接做分類，只抽 feature
###看notion的解釋
'''
CompactRouterFeatureEncoder:
  LLM prompt vector [B, H_llm]
  + BERT token memory [B, 2T, H_bert]
  -> cross-attention
  -> router feature [B, 2D]
'''
class CompactRouterFeatureEncoder(nn.Module):
    def __init__(self, llama_hidden_size: int, bert_hidden_size: int, router_dim: int):
        super().__init__()
        ###Question: 這邊的qkv的意思是什麼？
        self.q_proj = nn.Linear(llama_hidden_size, router_dim) ###query
        self.k_proj = nn.Linear(bert_hidden_size, router_dim) ###key
        self.v_proj = nn.Linear(bert_hidden_size, router_dim) ###value
        self.out_norm = nn.LayerNorm(router_dim * 2)

    def forward(self, llama_vec, bert_prev, bert_last, bert_attention_mask=None):
        router_dtype = self.q_proj.weight.dtype
        router_device = self.q_proj.weight.device
        llama_vec = llama_vec.to(device=router_device, dtype=router_dtype)
        bert_prev = bert_prev.to(device=router_device, dtype=router_dtype)
        bert_last = bert_last.to(device=router_device, dtype=router_dtype)

        q = self.q_proj(llama_vec).unsqueeze(1)
        mem = torch.cat([bert_prev, bert_last], dim=1)
        k = self.k_proj(mem)
        v = self.v_proj(mem)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        if bert_attention_mask is not None:
            mask = torch.cat([bert_attention_mask, bert_attention_mask], dim=1)
            mask = (mask == 0).unsqueeze(1).to(device=router_device)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).squeeze(1)
        feat = torch.cat([q.squeeze(1), ctx], dim=-1)
        return self.out_norm(feat)

####從 LLM 指定 layer 抓 first_vec / mid_vec
class PromptVectorExtractor(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        first_layer_idx: int,
        middle_layer_idx: int,
        pooling: str = "last_token",
        pooling_last_k: int = 4,
    ):
        super().__init__()
        self.model = model
        self.backbone_spec = infer_backbone_spec(model)
        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.pooling = str(pooling)
        self.pooling_last_k = int(pooling_last_k)
        self.cached_first = None
        self.cached_mid = None
        self._install_hooks()

    ###這裡的 .detach() 很重要：它表示這些 hidden states 只是拿來當 router feature，不讓梯度回傳去訓練 base LLM。
    ###Question: 這句話是什麼意思？
    def _install_hooks(self):
        def first_pre_hook(module, args):
            self.cached_first = args[0].detach()
            return None

        def mid_pre_hook(module, args):
            self.cached_mid = args[0].detach()
            return None

        layers = get_decoder_layers(self.model, spec=self.backbone_spec)
        ### LLM forward 跑到那兩層時，hook 會自動把進入該層 attention 前的 hidden states 存起來
        get_pre_attn_norm(layers[self.first_layer_idx], self.backbone_spec).register_forward_pre_hook(first_pre_hook)
        get_pre_attn_norm(layers[self.middle_layer_idx], self.backbone_spec).register_forward_pre_hook(mid_pre_hook)

    @staticmethod
    def gather_last_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx, :]

    @staticmethod
    def gather_mean_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    @staticmethod
    def gather_last_k_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor, k: int) -> torch.Tensor:
        k = max(int(k), 1)
        outputs = []
        lengths = attention_mask.sum(dim=1)
        for batch_idx in range(hidden_states.size(0)):
            valid_len = int(lengths[batch_idx].item())
            if valid_len <= 0:
                outputs.append(hidden_states[batch_idx, 0])
                continue
            start = max(0, valid_len - k)
            outputs.append(hidden_states[batch_idx, start:valid_len].mean(dim=0))
        return torch.stack(outputs, dim=0)

    '''
    抓到的 hidden states 原本 shape 大概是：

    [batch_size, seq_len, hidden_size]

    但 router 不直接吃整串 token，所以它會 pooling 成：

    [batch_size, hidden_size]

    pooling 有三種：

    last_token
    mean
    lastk_mean

    分別是：

    - last_token：取每筆 prompt 最後一個有效 token 的 hidden state。
    - mean：對所有有效 token 平均。
    - lastk_mean：對最後 k 個有效 token 平均。

    所以最後回傳：

    first_vec, mid_vec
    '''
    def gather_pooled(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last_token":
            return self.gather_last_valid(hidden_states, attention_mask)
        if self.pooling == "mean":
            return self.gather_mean_valid(hidden_states, attention_mask)
        if self.pooling == "lastk_mean":
            return self.gather_last_k_mean(hidden_states, attention_mask, self.pooling_last_k)
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.cached_first = None
        self.cached_mid = None
        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        if self.cached_first is None or self.cached_mid is None:
            raise RuntimeError("Failed to capture hidden states for router vectors.")
        first_vec = self.gather_pooled(self.cached_first, attention_mask)
        mid_vec = self.gather_pooled(self.cached_mid, attention_mask)
        return first_vec, mid_vec
