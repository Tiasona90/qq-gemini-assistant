"""
Image processing utilities for QQ Gemini Assistant.
Handles downloading images from OneBot/NapCat payloads,
high-fidelity Lanczos resampling to <= 1200px on long edge,
and CQ code / message segment extraction.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import httpx
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


async def download_image(source: str, timeout: float = 20.0) -> bytes:
    """
    Download or load image bytes from various OneBot v11 representations:
    1. HTTP/HTTPS URL
    2. base64://... or data:image/...;base64,...
    3. file://... or local file path
    """
    src = source.strip()
    if not src:
        raise ValueError("Empty image source")

    # 1. Base64 encoded format
    if src.startswith("base64://"):
        raw_b64 = src[len("base64://"):]
        return base64.b64decode(raw_b64)

    if src.startswith("data:image/") and ";base64," in src:
        raw_b64 = src.split(";base64,")[-1]
        return base64.b64decode(raw_b64)

    # 2. Local filesystem path or file:// URI
    if src.startswith("file:///"):
        local_path = Path(src[len("file:///"):])
        if local_path.exists():
            return local_path.read_bytes()
    elif src.startswith("file://"):
        local_path = Path(src[len("file://"):])
        if local_path.exists():
            return local_path.read_bytes()
    elif Path(src).exists() and Path(src).is_file():
        return Path(src).read_bytes()

    # 3. HTTP / HTTPS Network URL
    if src.startswith("http://") or src.startswith("https://"):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(src, headers=headers)
            resp.raise_for_status()
            return resp.content

    # If all above fail, raise error
    raise ValueError(f"Unsupported or inaccessible image source: {src[:60]}")


def compress_image(
    raw_bytes: bytes,
    max_long_edge: int = 1200,
    jpeg_quality: int = 90,
) -> Tuple[bytes, str, Tuple[int, int], Tuple[int, int], str]:
    """
    Intelligent 3-tier adaptive image processing & high-fidelity compression:
    - Tier 0 (long_edge <= 1200): Keep original pixels untouched (strategy='original').
    - Tier 1 (1200 < long_edge <= 1300, diff <= 100px): Direct center-crop (strategy='crop').
      Losslessly cuts off the extra <=100px from the long edge, retaining 100% original sensor pixels
      without any interpolation/resampling blur or ringing artifacts!
    - Tier 2 (1300 < long_edge <= 1600, mild downsampling scale >= 0.75): Bicubic interpolation (strategy='bicubic').
      Smooth natural transition without over-sharpening halos or screen moiré interference.
    - Tier 3 (long_edge > 1600, large scale downsampling scale < 0.75): Lanczos convolution (strategy='lanczos').
      High-density Sinc convolution filtering for extreme edge clarity and small character legibility on 2K/4K inputs.
    - Color: Alpha blended onto pure white; 4:4:4 chroma subsampling (subsampling=0) and Q90 JPEG export.

    Returns:
        tuple: (compressed_bytes, mime_type, original_size, final_size, strategy)
    """
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)

    orig_w, orig_h = img.size
    long_edge = max(orig_w, orig_h)
    strategy = "original"

    if long_edge <= max_long_edge:
        # Tier 0: Already within limit
        strategy = "original"
        new_w, new_h = orig_w, orig_h

    elif long_edge - max_long_edge <= 100:
        # Tier 1: Difference <= 100px -> Direct center-crop on long edge
        strategy = "crop"
        if orig_w >= orig_h:
            # Width is the long edge
            diff = orig_w - max_long_edge
            left = diff // 2
            right = left + max_long_edge
            top = 0
            bottom = orig_h
        else:
            # Height is the long edge
            diff = orig_h - max_long_edge
            top = diff // 2
            bottom = top + max_long_edge
            left = 0
            right = orig_w
        img = img.crop((left, top, right, bottom))
        new_w, new_h = img.size

    elif long_edge <= 1600:
        # Tier 2: Mild downscaling (1301px ~ 1600px) -> Bicubic interpolation
        strategy = "bicubic"
        scale = max_long_edge / float(long_edge)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    else:
        # Tier 3: Large scale downscaling (> 1600px, e.g. 2K/4K/raw) -> Lanczos convolution
        strategy = "lanczos"
        scale = max_long_edge / float(long_edge)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Convert RGBA/palette to clean RGB on white canvas
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img_rgba = img.convert("RGBA")
        background = Image.new("RGB", img_rgba.size, (255, 255, 255))
        background.paste(img_rgba, mask=img_rgba.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # High-fidelity JPEG export
    output_io = io.BytesIO()
    img.save(
        output_io,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,  # 4:4:4 full chroma preservation
        optimize=True,
    )
    compressed_data = output_io.getvalue()
    return compressed_data, "image/jpeg", (orig_w, orig_h), (new_w, new_h), strategy


def _unescape_cq(value: str, *, parameter: bool = False) -> str:
    """Decode exactly one CQ escape layer, with ampersand decoded last.

    CQ parameters also escape commas. Do not use HTML unescape here: arbitrary
    HTML entities and percent escapes can be literal parts of signed URLs.
    """
    if parameter:
        value = value.replace("&#44;", ",")
    return value.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "&")


def extract_message_and_images(event: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Read structured OneBot segments, falling back to decoded CQ strings.

    message and raw_message are two representations of the same message, not
    separate image lists. Prefer structured images so differing escape layers
    or URL representations cannot make one image appear twice.
    """
    texts: List[str] = []
    images: List[str] = []
    seen: set[str] = set()

    def add_image(ref: Any) -> None:
        if isinstance(ref, str):
            ref = ref.strip()
            if ref and ref not in seen:
                seen.add(ref)
                images.append(ref)

    msg_obj = event.get("message")
    if isinstance(msg_obj, list):
        for seg in msg_obj:
            if not isinstance(seg, dict):
                continue
            data = seg.get("data")
            if not isinstance(data, dict):
                continue
            if seg.get("type") == "text":
                text = data.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif seg.get("type") == "image":
                add_image(data.get("url") or data.get("file"))

    raw_msg = event.get("raw_message")
    if not isinstance(raw_msg, str) or not raw_msg:
        raw_msg = msg_obj if isinstance(msg_obj, str) else ""

    if raw_msg:
        if not images:
            for cq_param in re.findall(r"\[CQ:image,(.*?)\]", raw_msg, flags=re.DOTALL):
                # Split before unescaping, since a URL may contain encoded commas.
                params = {}
                for part in cq_param.split(","):
                    key, separator, value = part.partition("=")
                    if separator:
                        params[key.strip()] = _unescape_cq(value, parameter=True)
                add_image(params.get("url") or params.get("file"))
        if not texts:
            # Strip real CQ codes first so escaped literal [CQ:...] stays text.
            cleaned = _unescape_cq(re.sub(r"\[CQ:.*?\]", "", raw_msg)).strip()
            if cleaned:
                texts.append(cleaned)

    return "".join(texts).strip(), images
