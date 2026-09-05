from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_resolution.inference import (
    InferenceError,
    build_service_from_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDSR super-resolution inference on an image."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the enhanced PNG will be saved.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4],
        default=4,
        help="Super-resolution scale factor.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        service = build_service_from_environment()

        result = service.enhance(
            image_path=input_path,
            output_path=output_path,
            scale=args.scale,
        )

        print(json.dumps(result.to_dict(), indent=2))

        return 0

    except InferenceError as exc:
        print(f"Inference error: {exc}")
        return 1

    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
