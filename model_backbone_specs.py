####用來抽 decoder layers、替換指定 layer, 先寫出來放, 未來有可能要切model
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class BackboneSpec:
    family: str
    layers_attr_path: Tuple[str, ...]
    pre_attn_norm_name: str


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "llama": BackboneSpec(
        family="llama",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "mistral": BackboneSpec(
        family="mistral",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "gemma": BackboneSpec(
        family="gemma",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "gemma2": BackboneSpec(
        family="gemma2",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "gemma4": BackboneSpec(
        family="gemma4",
        layers_attr_path=("model", "language_model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "gemma4_text": BackboneSpec(
        family="gemma4_text",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "qwen2": BackboneSpec(
        family="qwen2",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "qwen3": BackboneSpec(
        family="qwen3",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
    "qwen2_moe": BackboneSpec(
        family="qwen2_moe",
        layers_attr_path=("model", "layers"),
        pre_attn_norm_name="input_layernorm",
    ),
}

FAMILY_ALIASES = {
    "qwen": "qwen2",
    "qwen3_5_text": "qwen3",
}


def infer_backbone_spec(model_or_config: Any) -> BackboneSpec:
    config = getattr(model_or_config, "config", model_or_config)
    model_type = str(getattr(config, "model_type", "llama")).lower()
    model_type = FAMILY_ALIASES.get(model_type, model_type)
    if model_type in BACKBONE_SPECS:
        return BACKBONE_SPECS[model_type]
    raise ValueError(
        f"Unsupported model_type={model_type!r}. "
        f"Known types: {sorted(BACKBONE_SPECS.keys())}"
    )


def _resolve_attr_path(root: Any, path: Iterable[str]) -> Any:
    cur = root
    for name in path:
        cur = getattr(cur, name)
    return cur


def get_decoder_layers(model: Any, spec: BackboneSpec | None = None):
    spec = spec or infer_backbone_spec(model)
    return _resolve_attr_path(model, spec.layers_attr_path)


def set_decoder_layer(model: Any, layer_idx: int, new_layer: Any, spec: BackboneSpec | None = None):
    layers = get_decoder_layers(model, spec=spec)
    layers[int(layer_idx)] = new_layer


def get_pre_attn_norm(layer: Any, spec: BackboneSpec) -> Any:
    return getattr(layer, spec.pre_attn_norm_name)


def get_hidden_size(model_or_config: Any) -> int:
    config = getattr(model_or_config, "config", model_or_config)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    language_model = getattr(model_or_config, "language_model", None)
    if language_model is not None:
        return get_hidden_size(language_model)
    raise AttributeError(f"Cannot infer hidden_size from {type(model_or_config).__name__}")
