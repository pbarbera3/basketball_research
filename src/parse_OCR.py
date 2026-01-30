import os
from tqdm import tqdm
import numpy as np
import argparse
import json
import re
import csv
import easyocr
import cv2

# src/parse_ocr.py
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ----------- Parsing helpers -----------


def clock_to_seconds(clock_str: str):
    """
    Convert NCAA game clock strings to seconds remaining in the period.
    Supports:
      - "MM:SS"  (e.g., "19:32")
      - "SS.D"   (e.g., "59.9")
      - "M:SS.D" or "0:59.9" (OCR sometimes returns this)
    Returns float seconds or None.
    """
    if not clock_str:
        return None
    s = clock_str.strip().replace(" ", "").replace(
        "•", "").replace("-", "").replace(";", ":")
    if s.startswith(":"):
        s = s[1:]
    s = s.replace("O", "0").replace("o", "0").replace(",", ":")

    # MM:SS
    if re.match(r"^\d{1,2}:\d{2}$", s):
        m, sec = s.split(":")
        return float(int(m) * 60 + int(sec))

    # SS.D
    if re.match(r"^\d{1,2}\.\d$", s):
        return float(s)

    # M:SS.D  or 0:59.9
    m = re.match(r"^(\d{1,2}):(\d{1,2})\.(\d)$", s)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        tenths = int(m.group(3))
        total = minutes * 60 + seconds + tenths / 10.0
        # For NCAA, under 1:00 display is effectively SS.D; but if OCR gave 0:59.9 this still works.
        return float(total)

    return None


def normalize_clock_text(raw: str):
    """Keep your normalization for writing raw text if you want it in CSV too."""
    if not raw:
        return None
    s = raw.strip().replace(" ", "").replace(
        "•", "").replace("-", "").replace(";", ":")
    if s.startswith(":"):
        s = s[1:]
    s = s.replace("O", "0").replace("o", "0").replace(",", ":")
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return s
    if re.match(r"^\d{1,2}\.\d$", s):
        return s
    m = re.match(r"^(\d{1,2}):(\d{1,2})\.(\d)$", s)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        tenths = int(m.group(3))
        # represent as SS.D when under 1 minute, else keep M:SS.D
        total_sec = minutes * 60 + seconds + tenths / 10.0
        if total_sec < 60:
            return f"{int(total_sec)}.{tenths}"
        return f"{minutes}:{seconds:02d}.{tenths}"
    return None

# ----------- Vision helpers -----------


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def get_frame_at_time(video_path: str, t_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000)
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return ret, frame, fps


def choose_frame_interactive(video_path: str, start_sec: float = 20.0):
    """
    Simple scrub UI to choose a good frame before ROI selection.

    Controls:
      A / Left  : -1s
      D / Right : +1s
      S / Down  : -10s
      W / Up    : +10s
      Enter     : confirm
      Q / Esc   : quit
    """
    t = float(start_sec)

    while True:
        ret, frame, fps = get_frame_at_time(video_path, t)
        if not ret:
            # If we went out of bounds, step back a bit
            t = max(0.0, t - 1.0)
            continue

        display = frame.copy()
        cv2.putText(
            display,
            f"Choose frame | t={t:.1f}s  (A/D=+-1s, W/S=+-10s, Enter=OK, Q=Quit)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Frame Selector", display)
        key = cv2.waitKey(0) & 0xFF

        # Enter
        if key in [13, 10]:
            cv2.destroyWindow("Frame Selector")
            return t, frame

        # Quit (q or esc)
        if key in [ord("q"), 27]:
            cv2.destroyWindow("Frame Selector")
            raise SystemExit("Frame selection cancelled by user.")

        # +/- 1 second
        if key in [ord("a"), 81]:   # left arrow often = 81
            t = max(0.0, t - 1.0)
        elif key in [ord("d"), 83]:  # right arrow often = 83
            t = t + 1.0

        # +/- 10 seconds
        elif key in [ord("s"), 84]:  # down arrow often = 84
            t = max(0.0, t - 10.0)
        elif key in [ord("w"), 82]:  # up arrow often = 82
            t = t + 10.0


def select_or_load_rois(video_path, rois_path, start_sec=20):
    if os.path.exists(rois_path):
        with open(rois_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Opening frame selector to find a frame where the clocks are visible...")
    chosen_t, frame = choose_frame_interactive(video_path, start_sec=start_sec)

    print("Select GAME CLOCK ROI and press ENTER.")
    r1 = cv2.selectROI("GAME CLOCK ROI", frame)
    cv2.destroyWindow("GAME CLOCK ROI")

    print("Select SHOT CLOCK ROI and press ENTER.")
    r2 = cv2.selectROI("SHOT CLOCK ROI", frame)
    cv2.destroyWindow("SHOT CLOCK ROI")

    rois = {
        "game_clock": [int(v) for v in r1],
        "shot_clock": [int(v) for v in r2],
        "start_sec": float(chosen_t)
    }
    os.makedirs(os.path.dirname(rois_path), exist_ok=True)
    with open(rois_path, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2)

    print(f"Saved ROIs to {rois_path}")
    return rois


def extract_ocr_map(
    video_path,
    output_csv,
    rois_path,
    start_sec=20,
    sample_fps=1,
    diff_threshold=2.0,
    use_gpu=True,
    expected_half_len_sec=20*60
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps / sample_fps))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    total_steps = max(1, (total_frames - start_frame) // frame_interval)

    rois = select_or_load_rois(video_path, rois_path, start_sec=start_sec)
    gx, gy, gw, gh = rois["game_clock"]
    sx, sy, sw, sh = rois["shot_clock"]

    reader = easyocr.Reader(["en"], gpu=use_gpu)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    prev_game_img = None
    prev_shot_img = None

    last_game_text = None
    last_shot_text = None

    last_game_sec = None
    last_shot_sec = None

    rows = []
    sampled = 0

    def changed(curr, prev):
        if prev is None:
            return True
        return float(np.mean(cv2.absdiff(curr, prev))) >= diff_threshold

    pbar = tqdm(total=total_steps, desc="OCR", unit="step")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_interval != 0:
            continue

        pbar.update(1)

        game_roi = preprocess_roi(frame[gy:gy+gh, gx:gx+gw])
        shot_roi = preprocess_roi(frame[sy:sy+sh, sx:sx+sw])

        game_changed = changed(game_roi, prev_game_img)
        shot_changed = changed(shot_roi, prev_shot_img)

        if game_changed:
            text = reader.readtext(
                game_roi, detail=0, allowlist="0123456789:.")
            norm = None
            sec = None
            for t in text:
                norm_t = normalize_clock_text(t)
                sec_t = clock_to_seconds(norm_t) if norm_t else None
                if sec_t is not None:
                    norm, sec = norm_t, sec_t
                    break
            if norm is not None:
                last_game_text = norm
                last_game_sec = sec
            prev_game_img = game_roi

        if shot_changed:
            text = reader.readtext(
                shot_roi, detail=0, allowlist="0123456789:.")
            norm = None
            sec = None
            for t in text:
                norm_t = normalize_clock_text(t)
                sec_t = clock_to_seconds(norm_t) if norm_t else None
                if sec_t is not None:
                    norm, sec = norm_t, sec_t
                    break
            if norm is not None:
                last_shot_text = norm
                last_shot_sec = sec
            prev_shot_img = shot_roi

        # Use previous game sec to detect jump; do it safely:
        # (we store prev_game_sec_for_period separately)
        # period tracking

        t_sec = frame_id / fps
        if last_game_sec is not None or last_shot_sec is not None:
            rows.append((
                t_sec,
                last_game_text,
                last_game_sec,
                last_shot_text,
                last_shot_sec
            ))

        sampled += 1

    pbar.close()

    cap.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_time_sec", "game_clock_text",
                   "game_clock_sec", "shot_clock_text", "shot_clock_sec"])
        w.writerows(rows)

    print(f"Saved OCR map to {output_csv} ({len(rows)} rows).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--game_id", required=True)
    p.add_argument("--start_sec", type=int, default=20)
    p.add_argument("--sample_fps", type=float, default=2.0)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    ocr_dir = os.path.join("data", "metadata", args.game_id)
    rois_path = os.path.join(ocr_dir, "rois.json")
    output_csv = os.path.join(ocr_dir, "ocr_map.csv")

    extract_ocr_map(
        video_path=args.video,
        output_csv=output_csv,
        rois_path=rois_path,
        start_sec=args.start_sec,
        sample_fps=args.sample_fps,
        use_gpu=args.gpu
    )
