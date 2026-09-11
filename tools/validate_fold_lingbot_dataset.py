import argparse
import os
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "wan_va"))
sys.path.insert(0, str(repo))

from configs import VA_CONFIGS
from dataset import MultiLatentLeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="fold_clothes_train_200")
    parser.add_argument("--expected-len", type=int, default=None)
    parser.add_argument("--dataset-init-worker", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    cfg = VA_CONFIGS[args.config_name]
    if os.getenv("DATASET_ROOT"):
        cfg.dataset_path = os.environ["DATASET_ROOT"]
        cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
        info_path = os.path.join(cfg.dataset_path, "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            cfg.obs_cam_keys = [
                key for key, value in info.get("features", {}).items()
                if value.get("dtype") == "video"
            ]
    cfg.rank = int(os.environ.get("RANK", 0))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.dataset_init_worker = args.dataset_init_worker
    cfg.load_worker = 0

    ds = MultiLatentLeRobotDataset(config=cfg, num_init_worker=args.dataset_init_worker)
    print(f"config={args.config_name}")
    print(f"dataset_len={len(ds)}")
    if args.expected_len is not None and len(ds) != args.expected_len:
        raise SystemExit(f"expected len {args.expected_len}, got {len(ds)}")
    if len(ds) == 0:
        raise SystemExit("dataset has no usable latent samples")

    sample = ds[min(args.sample_index, len(ds) - 1)]
    for key, value in sample.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"{key}: {shape}")

    expected_channels = [14, 15, 16, 17, 18, 19, 28, 21, 22, 23, 24, 25, 26, 29]
    if list(cfg.used_action_channel_ids) != expected_channels:
        raise SystemExit(
            f"incorrect action mapping: {cfg.used_action_channel_ids}, expected {expected_channels}"
        )
    actions = sample["actions"]
    action_mask = sample["actions_mask"]
    if actions.shape[0] != 30 or actions.shape[3] != 1:
        raise SystemExit(f"unexpected action shape: {tuple(actions.shape)}")
    if actions.shape[2] != cfg.action_per_frame or cfg.action_per_frame != 12:
        raise SystemExit(
            f"action_per_frame mismatch: sample={actions.shape[2]}, config={cfg.action_per_frame}"
        )
    active_channels = action_mask.any(dim=(1, 2, 3)).nonzero().flatten().tolist()
    if active_channels != sorted(expected_channels):
        raise SystemExit(
            f"incorrect active action channels: {active_channels}, expected {sorted(expected_channels)}"
        )
    inactive = [channel for channel in range(30) if channel not in expected_channels]
    if actions[inactive].count_nonzero().item() != 0:
        raise SystemExit("inactive action channels contain non-zero values")
    print(f"action_mapping={expected_channels}")
    print(f"action_norm_source={ds._datasets[0].norm_stat_source}")


if __name__ == "__main__":
    main()
