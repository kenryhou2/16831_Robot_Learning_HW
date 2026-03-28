#!/usr/bin/env python3
"""
view_tfevents.py

Launch TensorBoard for a TensorFlow / TensorBoard event file (.tfevents...)
or a log directory.

Examples:
    python view_tfevents.py /path/to/events.out.tfevents.123
    python view_tfevents.py /path/to/logdir
    python view_tfevents.py /path/to/events.out.tfevents.123 --port 6007
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def is_probably_event_file(path: Path) -> bool:
    name = path.name
    return ("tfevents" in name) or name.endswith(".tf.events")


def ensure_tensorboard_installed() -> None:
    try:
        import tensorboard  # noqa: F401
    except Exception:
        print("TensorBoard is not installed. Install it with:")
        print("  pip install tensorboard")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a .tfevents file or TensorBoard log directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6006, help="Port to use (default: 6006)")
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=5,
        help="TensorBoard reload interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    ensure_tensorboard_installed()

    input_path = Path(args.path).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}")
        sys.exit(1)

    temp_dir = None
    logdir_to_use = None

    try:
        if input_path.is_dir():
            logdir_to_use = str(input_path)
        elif input_path.is_file():
            if not is_probably_event_file(input_path):
                print(f"Warning: file name does not look like a TensorBoard event file: {input_path.name}")
                print("Attempting to launch anyway...")
            # TensorBoard wants a directory; create temp logdir and copy the file in
            temp_dir = tempfile.mkdtemp(prefix="tb_view_")
            run_dir = Path(temp_dir) / "run1"
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, run_dir / input_path.name)
            logdir_to_use = temp_dir
        else:
            print(f"Error: unsupported path type: {input_path}")
            sys.exit(1)

        url = f"http://{args.host}:{args.port}"
        print(f"Launching TensorBoard...")
        print(f"Logdir: {logdir_to_use}")
        print(f"Open:   {url}")

        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            logdir_to_use,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--reload_interval",
            str(args.reload_interval),
        ]

        # Runs until Ctrl+C
        subprocess.run(cmd, check=False)

    except KeyboardInterrupt:
        print("\nStopped TensorBoard.")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()