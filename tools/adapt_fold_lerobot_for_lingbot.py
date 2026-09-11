import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
ACTION_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6", "right_joint7",
]


def hardlink_or_copytree(src: Path, dst: Path) -> None:
    def copy_fn(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=copy_fn)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_action_quantiles(dst: Path, episodes):
    arrays = []
    for ep in tqdm(episodes, desc="read actions"):
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq = dst / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(pq, columns=["action"])
        arr = np.stack(df["action"].to_numpy()).astype(np.float32)
        if arr.shape[1] != 14:
            raise ValueError(f"{pq} action dim is {arr.shape[1]}, expected 14")
        arrays.append(arr)
    actions = np.concatenate(arrays, axis=0)
    q01_14 = np.quantile(actions, 0.01, axis=0).astype(float).tolist()
    q99_14 = np.quantile(actions, 0.99, axis=0).astype(float).tolist()
    q01_30 = [0.0] * 30
    q99_30 = [0.0] * 30
    for i in range(14):
        q01_30[14 + i] = q01_14[i]
        q99_30[14 + i] = q99_14[i]
    return {"q01": q01_30, "q99": q99_30, "q01_14": q01_14, "q99_14": q99_14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--action-text", default="fold clothes with both robot arms")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(src / "meta" / "info.json")
    if dst.exists():
        raise FileExistsError(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    hardlink_or_copytree(src, dst)

    episodes_path = dst / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    for ep in episodes:
        length = int(ep["length"])
        ep["action_config"] = [{
            "start_frame": 0,
            "end_frame": length,
            "action_text": args.action_text,
        }]
    write_jsonl(episodes_path, episodes)

    norm = compute_action_quantiles(dst, episodes)
    (dst / "meta" / "lingbot_action_norm.json").write_text(
        json.dumps(norm, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "src": str(src),
        "dst": str(dst),
        "episodes": len(episodes),
        "action_dim_original": 14,
        "action_dim_lingbot": 30,
        "used_action_channel_ids": list(range(14, 28)),
        "obs_cam_keys": CAM_KEYS,
        "action_names": ACTION_NAMES,
        "action_text": args.action_text,
        "note": "data/videos are hardlinked or copied; episodes.jsonl has action_config; latents still need extraction",
    }
    (dst / "meta" / "lingbot_adapt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
