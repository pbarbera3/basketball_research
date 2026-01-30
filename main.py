# main.py
import sys
import subprocess
import os
from pathlib import Path

# Fix OpenMP duplicate init (EasyOCR/torch/mkl on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ====== EDIT THESE ======
GAME_NAME = "Syracuse@Pitt"
ESPN_GAME_ID = "401724881"
VIDEO_PATH = r"data/full_games/Syracuse@Pitt.mp4"

USE_GPU_OCR = True
# =======================

PY = sys.executable


def run(cmd):
    print("\n>>", " ".join(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


def main():
    meta_dir = Path("data") / "metadata" / ESPN_GAME_ID
    pbp_path = meta_dir / "pbp.json"
    ocr_path = meta_dir / "ocr_map.csv"

    # 1) fetch ESPN pbp metadata (skip if exists)
    if not pbp_path.exists():
        run([PY, "src/fetch_data.py", "--espn_id",
            ESPN_GAME_ID, "--save_dir", "data/metadata"])
    else:
        print(f"\n✅ Skipping fetch_data (already exists): {pbp_path}")

    # 2) OCR map (skip if exists)
    if not ocr_path.exists():
        cmd = [PY, "src/parse_ocr.py", "--video",
               VIDEO_PATH, "--game_id", ESPN_GAME_ID]
        if USE_GPU_OCR:
            cmd.append("--gpu")
        run(cmd)
    else:
        print(f"\n✅ Skipping parse_ocr (already exists): {ocr_path}")

    # 3) generate 3PT clips
    run([
        PY, "src/generate_clips.py",
        "--game_id", ESPN_GAME_ID,
        "--video", VIDEO_PATH,
        "--game_name", GAME_NAME,
        "--pre_sec", "13",
        "--post_sec", "2",
        "--shot_clock_full", "30",
        "--tolerance", "2.0",
    ])

    print("\n✅ Pipeline completed for", ESPN_GAME_ID)


if __name__ == "__main__":
    main()
