import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
from tqdm import tqdm

from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae

DEFAULT_CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def load_episodes(root: Path):
    with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_frame_ids(start: int, end: int, ori_fps: int, target_fps: int):
    stride = max(1, round(ori_fps / target_fps))
    ids = list(range(start, end, stride))
    keep = ((len(ids) - 1) // 4) * 4 + 1
    ids = ids[:keep]
    if len(ids) < 5:
        raise ValueError(f"too few sampled frames: {len(ids)}")
    return ids, stride


def iter_overlapped_chunks(num_frames: int, chunk_size: int):
    if chunk_size < 5 or (chunk_size - 1) % 4 != 0:
        raise ValueError("--vae-chunk-frames must be 4n+1 and at least 5")
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        length = end - start
        if length < 5:
            break
        keep = ((length - 1) // 4) * 4 + 1
        end = start + keep
        yield start, end
        if end >= num_frames:
            break
        start = end - 1


def read_video_frames(path: Path, frame_ids, height: int, width: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {fid} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(frame).float())
    cap.release()
    return torch.stack(frames, dim=0).permute(3, 0, 1, 2).unsqueeze(0)


@torch.no_grad()
def encode_texts(model_root: Path, texts, device: str, dtype: torch.dtype):
    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(
        str(model_root / "text_encoder"), torch_dtype=dtype, torch_device=device
    )
    text_encoder.eval()
    out = {}
    for text in sorted(set(texts)):
        text_inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
        emb = torch.stack(
            [
                torch.cat([u[:v], u.new_zeros(512 - int(v), u.size(1))])
                for u, v in zip(emb, seq_lens)
            ],
            dim=0,
        )[0].cpu()
        out[text] = emb
    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def normalize_latents(latents, latents_mean, latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * latents_std).to(latents)


@torch.no_grad()
def encode_video_chunk(vae, video, device: str, dtype: torch.dtype):
    video = video / 255.0 * 2.0 - 1.0
    video = video.to(device=device, dtype=dtype)
    mu = vae.encode(video).latent_dist.mode()
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device)
    return normalize_latents(mu, latents_mean, 1.0 / latents_std)


def encode_camera_chunked(
    vae,
    video_path: Path,
    frame_ids,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    vae_chunk_frames: int,
):
    latent_chunks = []
    for chunk_idx, (s, e) in enumerate(iter_overlapped_chunks(len(frame_ids), vae_chunk_frames)):
        chunk_frame_ids = frame_ids[s:e]
        video = read_video_frames(video_path, chunk_frame_ids, height, width)
        latent = encode_video_chunk(vae, video, device, dtype)
        if chunk_idx > 0:
            latent = latent[:, :, 1:]
        latent_chunks.append(latent.cpu())
        del video, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mu_norm = torch.cat(latent_chunks, dim=2)
    latent = mu_norm[0].permute(1, 2, 3, 0).contiguous()
    f, h, w, c = latent.shape
    return latent.reshape(f * h * w, c), f, h, w


def write_empty_emb(root: Path, text_emb_cache):
    first = next(iter(text_emb_cache.values()))
    torch.save(torch.zeros_like(first), root / "empty_emb.pt")


def process_episode(args, ep, vae, text_emb_cache):
    root = Path(args.dataset)
    cfg = ep["action_config"][0]
    episode_index = int(ep["episode_index"])
    start = int(cfg["start_frame"])
    end = int(cfg["end_frame"])
    text = cfg["action_text"]
    frame_ids, stride = make_frame_ids(start, end, args.ori_fps, args.target_fps)
    if args.max_sampled_frames is not None and len(frame_ids) > args.max_sampled_frames:
        keep = ((args.max_sampled_frames - 1) // 4) * 4 + 1
        frame_ids = frame_ids[:keep]
    chunk = episode_index // 1000
    text_emb = text_emb_cache[text]

    for key in args.camera_keys:
        out_dir = root / "latents" / f"chunk-{chunk:03d}" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"episode_{episode_index:06d}_{start}_{end}.pth"
        if out_file.exists() and not args.overwrite:
            continue
        video_path = root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        latent, lf, lh, lw = encode_camera_chunked(
            vae,
            video_path,
            frame_ids,
            args.height,
            args.width,
            args.device,
            torch.bfloat16,
            args.vae_chunk_frames,
        )
        payload = {
            "latent": latent.to(torch.bfloat16),
            "latent_num_frames": int(lf),
            "latent_height": int(lh),
            "latent_width": int(lw),
            "video_num_frames": len(frame_ids),
            "video_height": args.height,
            "video_width": args.width,
            "text_emb": text_emb.to(torch.bfloat16),
            "text": text,
            "frame_ids": frame_ids,
            "start_frame": start,
            "end_frame": end,
            "fps": args.target_fps,
            "ori_fps": args.ori_fps,
        }
        torch.save(payload, out_file)
    return episode_index, len(frame_ids), stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--episode-end", type=int, default=None)
    ap.add_argument("--episode-index", type=int, default=None)
    ap.add_argument("--target-fps", type=int, default=10)
    ap.add_argument("--ori-fps", type=int, default=30)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae-chunk-frames", type=int, default=17)
    ap.add_argument("--max-sampled-frames", type=int, default=None)
    ap.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAM_KEYS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    model_root = Path(args.model)
    episodes = load_episodes(root)
    if args.episode_index is not None:
        episodes = [ep for ep in episodes if int(ep["episode_index"]) == args.episode_index]
    else:
        end = len(episodes) if args.episode_end is None else args.episode_end
        episodes = [ep for ep in episodes if args.episode_start <= int(ep["episode_index"]) < end]
    if not episodes:
        raise RuntimeError("no episodes selected")

    texts = [ep["action_config"][0]["action_text"] for ep in episodes]
    text_emb_cache = encode_texts(model_root, texts, args.device, torch.bfloat16)
    write_empty_emb(root, text_emb_cache)

    vae = load_vae(str(model_root / "vae"), torch_dtype=torch.bfloat16, torch_device=args.device)
    vae.eval()

    for ep in tqdm(episodes, desc="episodes"):
        ep_idx, sampled, stride = process_episode(args, ep, vae, text_emb_cache)
        print(f"episode={ep_idx} sampled_frames={sampled} stride={stride}")


if __name__ == "__main__":
    main()
