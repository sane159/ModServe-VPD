import io
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmm_serving.vpd.model_stages import run_decode_stage, run_prefill_stage, run_vision_stage
from lmm_serving.vpd.vision_cache import VisionFeatureLRU, encode_images_with_cache

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def make_image(seed_color, shape_color):
    img = Image.new("RGB", (384, 384), color=seed_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 320, 320], fill=shape_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue()


def prepare(processor, images, prompt):
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt").to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    return inputs


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    eos_id = processor.tokenizer.eos_token_id

    imgA, bytesA = make_image((220, 50, 50), (255, 220, 80))
    imgB, bytesB = make_image((40, 120, 220), (80, 240, 160))
    imgC, bytesC = make_image((120, 60, 190), (240, 120, 180))

    print("=== 1. Byte-budget LRU eviction check ===")
    tiny_lru = VisionFeatureLRU(max_bytes=200)
    tiny_lru.put(("k1",), torch.zeros(50, dtype=torch.bfloat16))
    tiny_lru.put(("k2",), torch.zeros(50, dtype=torch.bfloat16))
    assert tiny_lru.get(("k1",)) is not None
    tiny_lru.put(("k3",), torch.zeros(50, dtype=torch.bfloat16))
    assert tiny_lru.get(("k2",)) is None and tiny_lru.get(("k1",)) is not None and tiny_lru.get(("k3",)) is not None
    print("  LRU stats:", tiny_lru.stats())

    print("\n=== 2. Naive partial-hit BF16 drift vs Fixed cache ===")
    prompt = "Compare the shapes and colors in these two images."
    inp_AB = prepare(processor, [imgA, imgB], prompt)
    inp_AC = prepare(processor, [imgA, imgC], prompt)
    inp_A_only = prepare(processor, [imgA], "Describe this image.")

    ref_feats_AC = run_vision_stage(model, inp_AC["pixel_values"], inp_AC["pixel_attention_mask"])
    ref_ft, ref_kv = run_prefill_stage(model, inp_AC["input_ids"], inp_AC["attention_mask"], ref_feats_AC)
    ref_tokens = run_decode_stage(model, ref_ft, inp_AC["attention_mask"], ref_kv, 8, eos_id)[0].tolist()

    naive_cache = VisionFeatureLRU(max_bytes=256 * 1024 * 1024)
    _ = encode_images_with_cache(
        model, processor, [imgA], [bytesA], inp_A_only["pixel_values"], inp_A_only["pixel_attention_mask"],
        naive_cache, isolate_by_req_shape=False, pad_to_orig_batch=False,
    )
    naive_feats_AC = encode_images_with_cache(
        model, processor, [imgA, imgC], [bytesA, bytesC], inp_AC["pixel_values"], inp_AC["pixel_attention_mask"],
        naive_cache, isolate_by_req_shape=False, pad_to_orig_batch=False,
    )
    naive_diff = (naive_feats_AC.float() - ref_feats_AC.float()).abs().max().item()
    naive_nonzero = int((naive_feats_AC != ref_feats_AC).sum().item())
    print(f"  [Naive partial hit] max_abs_diff={naive_diff:.6f}, diff_elems={naive_nonzero}/{ref_feats_AC.numel()}")

    fixed_cache = VisionFeatureLRU(max_bytes=256 * 1024 * 1024)
    _ = encode_images_with_cache(
        model, processor, [imgA], [bytesA], inp_A_only["pixel_values"], inp_A_only["pixel_attention_mask"],
        fixed_cache, isolate_by_req_shape=True, pad_to_orig_batch=True,
    )
    _ = encode_images_with_cache(
        model, processor, [imgA, imgB], [bytesA, bytesB], inp_AB["pixel_values"], inp_AB["pixel_attention_mask"],
        fixed_cache, isolate_by_req_shape=True, pad_to_orig_batch=True,
    )
    fixed_feats_AC = encode_images_with_cache(
        model, processor, [imgA, imgC], [bytesA, bytesC], inp_AC["pixel_values"], inp_AC["pixel_attention_mask"],
        fixed_cache, isolate_by_req_shape=True, pad_to_orig_batch=True,
    )
    fixed_diff = (fixed_feats_AC.float() - ref_feats_AC.float()).abs().max().item()
    fixed_nonzero = int((fixed_feats_AC != ref_feats_AC).sum().item())

    fix_ft, fix_kv = run_prefill_stage(model, inp_AC["input_ids"], inp_AC["attention_mask"], fixed_feats_AC)
    fix_tokens = run_decode_stage(model, fix_ft, inp_AC["attention_mask"], fix_kv, 8, eos_id)[0].tolist()

    print(f"  [Fixed partial hit] max_abs_diff={fixed_diff:.6f}, diff_elems={fixed_nonzero}/{ref_feats_AC.numel()}")
    print(f"  Ref tokens  : {ref_tokens}")
    print(f"  Fixed tokens: {fix_tokens}")
    print("  Fixed cache stats:", fixed_cache.stats())

    assert fixed_diff == 0.0 and fixed_nonzero == 0, f"Expected exact 0.0 BF16 diff, got {fixed_diff}"
    assert fix_tokens == ref_tokens, "Token mismatch in fixed cache mode!"
    print("\nPASS: Milestone 4 verified (LRU eviction + BF16 drift fixed to 0.0)!")


if __name__ == "__main__":
    main()
