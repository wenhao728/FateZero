import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video


def format_video_pt(video_pt: torch.Tensor) -> torch.Tensor:
    assert video_pt.dim() == 4 and video_pt.size(1) == 3, \
        f"Video should have shape (T, 3, H, W), but got {video_pt.size()}"
    assert video_pt.max() <= 1.0 and video_pt.min() >= 0.0, \
        f"Video should be in range [0, 1], but got {video_pt.min()} - {video_pt.max()}"

    video_pt = (video_pt.cpu() * 255).to(torch.uint8).permute(0, 2, 3, 1)  # (T, H, W, C)
    return video_pt


def format_video_pil(video_pil: List[Image.Image]) -> torch.Tensor:
    for i in range(len(video_pil)):
        video_pil[i] = np.array(video_pil[i].convert('RGB'))  # (H, W, C)

    return torch.from_numpy(np.stack(video_pil, axis=0))  # (T, H, W, C)


def format_video_np(video_np: np.ndarray) -> torch.Tensor:
    assert video_np.ndim == 4 and video_np.shape[1] == 3, \
        f"Video should have shape (T, 3, H, W), but got {video_np.shape}"
    assert video_np.max() <= 1 and video_np.min() >= 0, \
        f"Video should be in range [0, 1], but got {video_np.min()} - {video_np.max()}"

    return torch.from_numpy((video_np * 255).astype(np.uint8)).permute(0, 2, 3, 1)  # (T, H, W, C)


def save_video(
    filename: os.PathLike, 
    video: Union[np.ndarray, torch.Tensor, List[Image.Image]], 
    fps: int,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(video, torch.Tensor):
        video = format_video_pt(video)
    elif isinstance(video, list) and isinstance(video[0], Image.Image):
        video = format_video_pil(video)
    elif isinstance(video, np.ndarray):
        video = format_video_np(video)
    else:
        raise ValueError(f'Unsupported video type: {type(video)}')

    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }
    write_video(filename.with_suffix('.mp4'), video, fps, video_codec=video_codec, options=video_options)
