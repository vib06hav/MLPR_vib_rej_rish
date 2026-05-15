from __future__ import annotations

import argparse
import json

from detection.config import DetectionConfig
from detection.engine import save_json
from detection.verify import run_all_verifications


def parse_args():
    parser = argparse.ArgumentParser(description="Verification-only checks for the RetinaNet BDD extension.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DetectionConfig.from_args(
        image_size=args.image_size,
        max_eval_samples=args.max_eval_samples,
    )
    cfg.ensure_output_dirs()
    payload = run_all_verifications(cfg)
    output_path = args.output_json or str(cfg.verification_dir / "verification_report.json")
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
