"""Recreate the media from the exact experiment source, preserving the workspace.

Install project dependencies and imageio-ffmpeg==0.8.1 first.
Run: PDSTL_DEVICE=cpu python deck_media/reproduce.py
"""

import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "12b3bd7987faab130ebcf73120aeddbf77f17085"


def main():
    archive = subprocess.check_output(
        ["git", "archive", COMMIT], cwd=HERE.parent
    )
    with tempfile.TemporaryDirectory(prefix="pdstl-deck-main-") as source:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        env = {
            **os.environ,
            "PDSTL_DEVICE": "cpu",
            "MPLCONFIGDIR": "/tmp/pdstl-mpl",
        }
        for script in ("run_reach_avoid.py", "render_media.py"):
            subprocess.run(
                [sys.executable, str(HERE / script), "--source", source],
                check=True,
                env=env,
            )
        subprocess.run(
            [sys.executable, str(HERE / "validate_media.py")],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
