#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BAD_EPISODES = {102, 1310, 1416, 1482, 1781, 1979}
ACTION_CHANNELS = [14, 15, 16, 17, 18, 19, 28, 21, 22, 23, 24, 25, 26, 29]
ACTION_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_gripper",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_gripper",
]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_symlink(path, target):
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--expected-len", type=int, default=2846)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).absolute()
    if dst.exists():
        raise FileExistsError(dst)

    episodes = [
        row for row in read_jsonl(src / "meta" / "episodes.jsonl")
        if int(row["episode_index"]) not in BAD_EPISODES
    ]
    if len(episodes) != args.expected_len:
        raise RuntimeError(f"expected {args.expected_len} episodes, got {len(episodes)}")

    dst.mkdir(parents=True)
    (dst / "meta").mkdir()
    for name in ["data", "videos", "latents"]:
        replace_symlink(dst / name, (src / name).resolve())
    if (src / "empty_emb.pt").is_file():
        replace_symlink(dst / "empty_emb.pt", (src / "empty_emb.pt").resolve())

    kept_ids = {int(row["episode_index"]) for row in episodes}
    arrays = []
    total_frames = 0
    for number, episode in enumerate(episodes, 1):
        episode_id = int(episode["episode_index"])
        parquet = dst / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
        frame = pd.read_parquet(parquet, columns=["action"])
        action = np.stack(frame.action.to_numpy()).astype(np.float32)
        expected_frames = int(episode["length"])
        if action.shape != (expected_frames, 14):
            raise ValueError(f"{parquet}: action shape {action.shape}, expected {(expected_frames, 14)}")
        if not np.isfinite(action).all():
            raise ValueError(f"{parquet}: non-finite action")
        if np.abs(action[:, :6]).max() > 4 or np.abs(action[:, 7:13]).max() > 4:
            raise ValueError(f"{parquet}: joint action outside [-4, 4]")
        if ((action[:, 6] < -0.02) | (action[:, 6] > 0.2)).any():
            raise ValueError(f"{parquet}: left gripper outside [-0.02, 0.2]")
        if ((action[:, 13] < -0.02) | (action[:, 13] > 0.2)).any():
            raise ValueError(f"{parquet}: right gripper outside [-0.02, 0.2]")
        arrays.append(action)
        total_frames += expected_frames
        if number % 500 == 0:
            print(f"validated episodes={number} frames={total_frames}", flush=True)

    actions = np.concatenate(arrays, axis=0)
    q01_14 = np.quantile(actions, 0.01, axis=0).astype(float).tolist()
    q99_14 = np.quantile(actions, 0.99, axis=0).astype(float).tolist()
    q01_30 = [0.0] * 30
    q99_30 = [0.0] * 30
    for source_id, channel_id in enumerate(ACTION_CHANNELS):
        q01_30[channel_id] = q01_14[source_id]
        q99_30[channel_id] = q99_14[source_id]

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = len(episodes) * sum(
        feature.get("dtype") == "video" for feature in info["features"].values()
    )
    info["splits"] = {"train": f"0:{len(episodes)}"}
    (dst / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(dst / "meta" / "episodes.jsonl", episodes)

    for name in ["episodes_stats.jsonl", "tasks.jsonl"]:
        source = src / "meta" / name
        if not source.is_file():
            continue
        rows = read_jsonl(source)
        if name == "episodes_stats.jsonl":
            rows = [row for row in rows if int(row["episode_index"]) in kept_ids]
        write_jsonl(dst / "meta" / name, rows)

    norm = {
        "q01": q01_30,
        "q99": q99_30,
        "q01_14": q01_14,
        "q99_14": q99_14,
        "used_action_channel_ids": ACTION_CHANNELS,
    }
    (dst / "meta" / "lingbot_action_norm.json").write_text(
        json.dumps(norm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "source": str(src),
        "destination": str(dst),
        "episodes": len(episodes),
        "frames": total_frames,
        "removed_episode_ids": sorted(BAD_EPISODES),
        "action_names": ACTION_NAMES,
        "used_action_channel_ids": ACTION_CHANNELS,
        "action_per_frame": 12,
    }
    (dst / "meta" / "lingbot_adapt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
