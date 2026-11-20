import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPooling


def pack_kv_cache(past_key_values) -> torch.Tensor:
    """将多层 KV 打包为单个连续张量: [L, 2, B, H, S, D]"""
    if hasattr(past_key_values, "layers"):
        per_layer = [torch.stack([layer.keys, layer.values], dim=0) for layer in past_key_values.layers]
    elif hasattr(past_key_values, "key_cache"):
        per_layer = [
            torch.stack([k, v], dim=0)
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
        ]
    else:
        per_layer = [torch.stack([k, v], dim=0) for k, v in past_key_values]
    return torch.stack(per_layer, dim=0).contiguous()


def unpack_kv_cache(packed_kv: torch.Tensor) -> DynamicCache:
    """从 [L, 2, B, H, S, D] 还原为 DynamicCache"""
    cache = DynamicCache()
    for layer_idx in range(packed_kv.shape[0]):
        k = packed_kv[layer_idx, 0].contiguous()
        v = packed_kv[layer_idx, 1].contiguous()
        cache.update(k, v, layer_idx)
    return cache


@torch.inference_mode()
def run_vision_stage(model, pixel_values, pixel_attention_mask) -> torch.Tensor:
    feats = model.model.get_image_features(
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
    )
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        feats = feats.pooler_output
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), dim=0)
    return feats.contiguous()


@torch.inference_mode()
def run_prefill_stage(model, input_ids, attention_mask, image_hidden_states):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_encoder_outputs={"image": BaseModelOutputWithPooling(pooler_output=image_hidden_states)},
        use_cache=True,
    )
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    packed_kv = pack_kv_cache(outputs.past_key_values)
    return first_token, packed_kv


@torch.inference_mode()
def run_decode_stage(model, first_token, attention_mask, packed_kv, max_new_tokens: int, eos_token_id: int):
    past_key_values = unpack_kv_cache(packed_kv)
    generated = [first_token]
    cur_token = first_token
    cur_mask = attention_mask

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and cur_token.item() == eos_token_id:
            break
        cur_mask = torch.cat(
            [cur_mask, torch.ones((cur_mask.shape[0], 1), device=cur_mask.device, dtype=cur_mask.dtype)],
            dim=1,
        )
        outputs = model(
            input_ids=cur_token,
            attention_mask=cur_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        cur_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated.append(cur_token)

    return torch.cat(generated, dim=1)


def prune_model_for_stage(model, role: str):
    """Drop unused submodules per stage so each worker only keeps its own weights."""
    import gc
    before_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    if role == "V":
        model.model.text_model = None
        model.lm_head = None
    elif role in ("P", "D"):
        model.model.vision_model = None
        model.model.connector = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    return before_mb, after_mb
