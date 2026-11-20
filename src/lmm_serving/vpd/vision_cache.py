from collections import OrderedDict
import hashlib
import io
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch

from lmm_serving.vpd.model_stages import run_vision_stage

_TILE_COUNT_BY_SIZE: Dict[Tuple[int, int], int] = {}
_IMG_EXPANSION_BY_SIZE: Dict[Tuple[int, int], str] = {}


class VisionFeatureLRU:
    """Content-addressed visual feature LRU cache with byte-budget eviction."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._store: "OrderedDict[Tuple, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def content_hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def make_key(
        img_hash: str,
        num_images: int,
        total_tiles: int,
        tile_offset: int,
        isolate_by_req_shape: bool = True,
    ) -> Tuple:
        if isolate_by_req_shape:
            return (img_hash, num_images, total_tiles, tile_offset)
        return (img_hash,)

    def has(self, key: Tuple) -> bool:
        return key in self._store

    def get(self, key: Tuple) -> Optional[torch.Tensor]:
        feat = self._store.get(key)
        if feat is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return feat

    def put(self, key: Tuple, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().contiguous()
        nbytes = tensor.element_size() * tensor.nelement()
        if nbytes > self.max_bytes:
            return

        if key in self._store:
            old = self._store.pop(key)
            self.current_bytes -= old.element_size() * old.nelement()

        while self._store and (self.current_bytes + nbytes > self.max_bytes):
            _, evicted = self._store.popitem(last=False)
            self.current_bytes -= evicted.element_size() * evicted.nelement()
            self.evictions += 1

        self._store[key] = tensor
        self.current_bytes += nbytes

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": (self.hits / total) if total > 0 else 0.0,
        }


def _ensure_size_metadata(processor, img: Image.Image) -> Tuple[int, str]:
    sz = img.size
    n_tiles = _TILE_COUNT_BY_SIZE.get(sz)
    expansion = _IMG_EXPANSION_BY_SIZE.get(sz)
    if n_tiles is None or expansion is None:
        single = processor(text="<image>", images=[img], return_tensors="pt", add_special_tokens=False)
        n_tiles = int(single["pixel_values"].shape[1])
        expansion = processor.tokenizer.decode(single["input_ids"][0], skip_special_tokens=False)
        _TILE_COUNT_BY_SIZE[sz] = n_tiles
        _IMG_EXPANSION_BY_SIZE[sz] = expansion
    return n_tiles, expansion


def compute_tile_slices(processor, images, total_tiles: int) -> List[Tuple[int, int]]:
    slices: List[Tuple[int, int]] = []
    offset = 0
    for img in images:
        n_tiles, _ = _ensure_size_metadata(processor, img)
        slices.append((offset, offset + n_tiles))
        offset += n_tiles
    assert offset == total_tiles, f"Tile count mismatch: {offset} != {total_tiles}"
    return slices


def try_full_hit_fast_path(
    processor,
    prompt_text: str,
    images_bytes: List[bytes],
    cache: Optional[VisionFeatureLRU],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Fast path when every image in the request hits the LRU cache:
    skips pixel decoding, CPU image_processor tiling, and GPU vision encoder completely.
    """
    if cache is None or not images_bytes:
        return None

    sizes: List[Tuple[int, int]] = []
    for b in images_bytes:
        with Image.open(io.BytesIO(b)) as im:
            sz = im.size
        if sz not in _TILE_COUNT_BY_SIZE or sz not in _IMG_EXPANSION_BY_SIZE:
            return None
        sizes.append(sz)

    num_images = len(images_bytes)
    tile_counts = [_TILE_COUNT_BY_SIZE[sz] for sz in sizes]
    total_tiles = sum(tile_counts)

    offsets = []
    cur = 0
    for n in tile_counts:
        offsets.append(cur)
        cur += n

    keys = [
        VisionFeatureLRU.make_key(
            VisionFeatureLRU.content_hash(images_bytes[i]),
            num_images=num_images,
            total_tiles=total_tiles,
            tile_offset=offsets[i],
            isolate_by_req_shape=True,
        )
        for i in range(num_images)
    ]

    if not all(cache.has(k) for k in keys):
        return None

    # All images hit: touch LRU, concat cached features, and tokenize expanded text only
    feats = [cache.get(k) for k in keys]
    parts = prompt_text.split("<image>")
    expanded = "".join(
        parts[i] + (_IMG_EXPANSION_BY_SIZE[sizes[i]] if i < len(sizes) else "")
        for i in range(len(parts))
    )

    tok = processor.tokenizer(expanded, return_tensors="pt")
    return tok["input_ids"], tok["attention_mask"], torch.cat(feats, dim=0).contiguous()


@torch.inference_mode()
def encode_images_with_cache(
    model,
    processor,
    images,
    images_bytes: List[bytes],
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    cache: Optional[VisionFeatureLRU],
    isolate_by_req_shape: bool = True,
    pad_to_orig_batch: bool = True,
) -> torch.Tensor:
    if cache is None:
        return run_vision_stage(model, pixel_values, pixel_attention_mask)

    num_images = len(images)
    total_tiles = int(pixel_values.shape[1])
    tile_slices = compute_tile_slices(processor, images, total_tiles)
    tokens_per_tile = (
        model.config.vision_config.image_size // model.config.vision_config.patch_size // model.config.scale_factor
    ) ** 2

    img_hashes = [VisionFeatureLRU.content_hash(b) for b in images_bytes]
    keys = [
        VisionFeatureLRU.make_key(
            img_hashes[i],
            num_images=num_images,
            total_tiles=total_tiles,
            tile_offset=tile_slices[i][0],
            isolate_by_req_shape=isolate_by_req_shape,
        )
        for i in range(num_images)
    ]

    cached_parts: List[Optional[torch.Tensor]] = [cache.get(k) for k in keys]
    miss_indices = [i for i, feat in enumerate(cached_parts) if feat is None]

    if not miss_indices:
        return torch.cat(cached_parts, dim=0).contiguous()

    if len(miss_indices) == num_images:
        full_feats = run_vision_stage(model, pixel_values, pixel_attention_mask)
        for i, (s, e) in enumerate(tile_slices):
            part = full_feats[s * tokens_per_tile : e * tokens_per_tile].contiguous()
            cache.put(keys[i], part)
        return full_feats

    if pad_to_orig_batch:
        padded_pv = torch.ones_like(pixel_values)
        padded_mask = pixel_attention_mask.clone()
        for i in miss_indices:
            s, e = tile_slices[i]
            padded_pv[:, s:e] = pixel_values[:, s:e]
            padded_mask[:, s:e] = pixel_attention_mask[:, s:e]

        padded_feats = run_vision_stage(model, padded_pv, padded_mask)
        for i in miss_indices:
            s, e = tile_slices[i]
            part = padded_feats[s * tokens_per_tile : e * tokens_per_tile].contiguous()
            cache.put(keys[i], part)
            cached_parts[i] = part
    else:
        miss_pv = torch.cat([pixel_values[:, tile_slices[i][0] : tile_slices[i][1]] for i in miss_indices], dim=1)
        miss_mask = torch.cat(
            [pixel_attention_mask[:, tile_slices[i][0] : tile_slices[i][1]] for i in miss_indices], dim=1
        )
        miss_feats = run_vision_stage(model, miss_pv, miss_mask)
        cur = 0
        for i in miss_indices:
            n_tiles = tile_slices[i][1] - tile_slices[i][0]
            n_tok = n_tiles * tokens_per_tile
            part = miss_feats[cur : cur + n_tok].contiguous()
            cur += n_tok
            cache.put(keys[i], part)
            cached_parts[i] = part

    return torch.cat(cached_parts, dim=0).contiguous()
