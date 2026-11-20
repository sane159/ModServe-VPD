import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPooling

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
MAX_NEW_TOKENS = 8


def pack_kv_cache(past_key_values) -> torch.Tensor:
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


def main():
    print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
    ).to(DEVICE).eval()

    img1 = Image.new("RGB", (384, 384), color=(220, 40, 40))
    img2 = Image.new("RGB", (384, 384), color=(40, 120, 220))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Describe the colors of these two images briefly."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img1, img2], return_tensors="pt").to(DEVICE)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    with torch.inference_mode():
        golden_full = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    golden_tokens = golden_full[:, prompt_len:]

    vision_feats = run_vision_stage(model, inputs["pixel_values"], inputs["pixel_attention_mask"])
    first_token, packed_kv = run_prefill_stage(
        model, inputs["input_ids"], inputs["attention_mask"], vision_feats
    )
    vpd_tokens = run_decode_stage(
        model,
        first_token,
        inputs["attention_mask"],
        packed_kv,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    print(f"vision_feats shape : {tuple(vision_feats.shape)}, dtype={vision_feats.dtype}")
    print(f"packed_kv shape    : {tuple(packed_kv.shape)}, dtype={packed_kv.dtype}")
    print(f"Golden token IDs   : {golden_tokens[0].tolist()}")
    print(f"VPD    token IDs   : {vpd_tokens[0].tolist()}")
    print("Golden text        :", processor.batch_decode(golden_tokens, skip_special_tokens=True)[0])
    print("VPD    text        :", processor.batch_decode(vpd_tokens, skip_special_tokens=True)[0])

    assert torch.equal(golden_tokens, vpd_tokens), "Mismatch between monolithic and VPD tokens!"
    print("PASS: 单模型与 VPD 三阶段逐 Token 完全一致！")


if __name__ == "__main__":
    main()
