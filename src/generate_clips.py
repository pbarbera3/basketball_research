import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Helpers: OCR period fixing
# ----------------------------
def make_period_calc_from_ocr_period(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a robust period_calc from OCR period column:
    - fill missing
    - force int
    - enforce non-decreasing over time (fixes occasional OCR glitches)
    """
    df = ocr_df.sort_values("video_time_sec").copy()
    p = pd.to_numeric(df["period"], errors="coerce").ffill().fillna(
        1).astype(int).to_numpy()
    p = np.maximum.accumulate(p)  # enforce monotonic non-decreasing
    df["period_calc"] = p
    return df


def add_period_from_clock_jumps(
    ocr_df: pd.DataFrame,
    jump_thresh=900.0,
    hi=1000.0,
    lo=180.0,
    min_separation_sec=30.0
):
    """
    Robust period reconstruction from game clock resets.

    We increment period when we see a *large upward jump* in game clock
    (end-of-half -> start-of-half), e.g. 4.6 -> 1188.

    lo=300 makes it work even if OCR didn't capture the last ~2 minutes.
    min_separation_sec prevents double-counting due to OCR noise.
    """
    df = ocr_df.sort_values("video_time_sec").copy()

    gc = pd.to_numeric(df["game_clock_sec"],
                       errors="coerce").to_numpy(dtype=float)
    vt = pd.to_numeric(df["video_time_sec"],
                       errors="coerce").to_numpy(dtype=float)

    period = 1
    periods = []
    prev_gc = None
    last_bump_time = -1e9

    for i in range(len(df)):
        cur_gc = gc[i]
        cur_t = vt[i] if not np.isnan(vt[i]) else (vt[i-1] if i > 0 else 0.0)

        if np.isnan(cur_gc):
            periods.append(period)
            continue

        if prev_gc is not None:
            jump = cur_gc - prev_gc

            # halftime reset-like jump
            if (
                jump > jump_thresh
                and cur_gc >= hi
                and prev_gc <= lo
                and prev_gc < 5.0                 # 👈 NEW: explicit near-zero
                and (cur_t - last_bump_time) >= min_separation_sec
            ):
                period += 1
                last_bump_time = cur_t

        periods.append(period)
        prev_gc = cur_gc

    df["period_calc"] = periods
    return df


# ----------------------------
# Helpers: parsing clocks/names
# ----------------------------

def clock_to_seconds_mmss(clock_str: str):
    """ESPN pbp clock is MM:SS (no tenths). Returns int seconds."""
    if not clock_str:
        return None
    s = clock_str.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def safe_name(s: str):
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-@]+", "", s)
    return s[:180]


def parse_shooter_name_from_text(text: str):
    """
    ESPN pbp text usually like:
      "Cooper Flagg made Three Point Jumper."
      "RJ Davis missed Three Point Jumper."
    """
    if not text:
        return None
    m = re.match(r"^(.*?)\s+(made|missed)\b",
                 text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


# ----------------------------
# ESPN JSON loading
# ----------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def http_get_json(url: str, timeout=15):
    """
    ESPN endpoints are publicly accessible; use a UA to avoid some blocks.
    """
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; BasketballDatasetBot/1.0)",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8", errors="replace"))


# ----------------------------
# Player stats fetching + parsing
# ----------------------------

def _collect_stats_from_payload(payload):
    """
    Try to normalize various ESPN statistics JSON formats into a {name: value} dict.
    """
    stats_map = {}

    if not isinstance(payload, dict):
        return stats_map

    # Common structures:
    # 1) payload["splits"]["categories"][i]["stats"] = [{"name":..., "value":...}, ...]
    # 2) payload["categories"][i]["stats"] = ...
    cats = None
    if isinstance(payload.get("splits"), dict) and isinstance(payload["splits"].get("categories"), list):
        cats = payload["splits"]["categories"]
    elif isinstance(payload.get("categories"), list):
        cats = payload["categories"]

    if cats:
        for c in cats:
            if not isinstance(c, dict):
                continue
            stats = c.get("stats")
            if isinstance(stats, list):
                for s in stats:
                    if isinstance(s, dict) and "name" in s and "value" in s:
                        stats_map[str(s["name"])] = s["value"]
            elif isinstance(stats, dict):
                # sometimes already a mapping
                for k, v in stats.items():
                    stats_map[str(k)] = v

    # Also try a flat "statistics"/"stats" dict if present
    for k in ["statistics", "stats"]:
        if isinstance(payload.get(k), dict):
            for kk, vv in payload[k].items():
                stats_map[str(kk)] = vv

    return stats_map


def _pick_3pt_from_statmap(statmap: dict):
    """
    Try to extract:
      - 3PM (made)
      - 3PA (attempted)
      - 3P% (pct)
    from many possible ESPN naming conventions.
    """
    made_keys = {
        "threePointFieldGoalsMade",
        "threePointFieldGoals",
        "fg3Made",
        "threePointMade",
        "3ptMade",
        "threePointFGMade",
        "threePointMadeFieldGoals",
    }
    att_keys = {
        "threePointFieldGoalsAttempted",
        "threePointFieldGoalAttempts",
        "fg3Attempted",
        "threePointAttempts",
        "3ptAttempted",
        "3ptAtt",
        "threePointFGA",
    }
    pct_keys = {
        "threePointFieldGoalPct",
        "threePointPercentage",
        "fg3Pct",
        "threePointPct",
        "3ptPct",
        "threePointFGPct",
    }

    def first_present(keys):
        for k in keys:
            if k in statmap:
                return statmap[k]
        return None

    made = first_present(made_keys)
    att = first_present(att_keys)
    pct = first_present(pct_keys)

    # Coerce to numeric if possible
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    made_f = to_float(made)
    att_f = to_float(att)
    pct_f = to_float(pct)

    if pct_f is None and made_f is not None and att_f not in (None, 0.0):
        pct_f = made_f / att_f

    return made_f, att_f, pct_f


def fetch_player_name_and_3pt_stats(athlete_id: str, season_year: int, season_type: int = 2):
    """
    Best-effort: try a few ESPN endpoints.
    Returns:
      (player_name, made, att, pct)
    Any missing values -> None.
    """
    athlete_id = str(athlete_id)

    # Try athlete profile (usually has displayName/fullName)
    athlete_name = None

    athlete_urls = [
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/{athlete_id}",
        # sometimes this works too:
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/athletes/{athlete_id}",
    ]

    for url in athlete_urls:
        try:
            a = http_get_json(url)
            athlete_name = a.get("displayName") or a.get(
                "fullName") or a.get("shortName")
            if athlete_name:
                break
        except Exception:
            continue

    # Try stats endpoint(s)
    stat_payloads = []
    stats_urls = [
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/{season_year}/types/{season_type}/athletes/{athlete_id}/statistics",
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/{athlete_id}/statistics",
    ]

    for url in stats_urls:
        try:
            stat_payloads.append(http_get_json(url))
        except Exception:
            continue

    made = att = pct = None
    for payload in stat_payloads:
        statmap = _collect_stats_from_payload(payload)
        m, a, p = _pick_3pt_from_statmap(statmap)
        if any(v is not None for v in [m, a, p]):
            made, att, pct = m, a, p
            break

    return athlete_name, made, att, pct


# ----------------------------
# PBP: extract 3PT plays
# ----------------------------

def extract_three_pt_plays(pbp_json):
    plays = pbp_json.get("plays", [])

    # team map (id -> displayName)
    comps = pbp_json.get("header", {}).get("competitions", [])
    team_map = {}
    if comps:
        for c in comps[0].get("competitors", []):
            tid = str(c["team"]["id"])
            team_map[tid] = {
                "name": c["team"]["displayName"],
                "homeAway": c.get("homeAway"),
            }

    out = []
    for p in plays:
        if p.get("pointsAttempted") != 3:
            continue

        period = p.get("period", {}).get("number", None)
        clock_txt = p.get("clock", {}).get("displayValue", None)
        clock_sec = clock_to_seconds_mmss(clock_txt)

        team_id = str(p.get("team", {}).get("id", "")) if p.get("team") else ""
        team_name = team_map.get(team_id, {}).get("name", team_id)

        made = bool(p.get("scoringPlay")) and int(
            p.get("scoreValue") or 0) == 3

        participants = p.get("participants") or []
        shooter_id = None
        if len(participants) >= 1 and isinstance(participants[0], dict):
            shooter = participants[0].get("athlete")
            if isinstance(shooter, dict) and shooter.get("id"):
                shooter_id = str(shooter["id"])

        shooter_name_guess = parse_shooter_name_from_text(p.get("text", ""))

        out.append({
            "play_id": p.get("id"),
            "sequence": p.get("sequenceNumber"),
            "period": period,
            "clock_text": clock_txt,
            "clock_sec": clock_sec,
            "team_id": team_id,
            "team_name": team_name,
            "made": made,
            "text": p.get("text", ""),
            "shooter_id": shooter_id,
            "shooter_name_guess": shooter_name_guess,
        })

    return out


# ----------------------------
# Matching: PBP -> OCR -> video time
# ----------------------------
def normalize_mmss_text(s: str):
    if not s or s == "nan":
        return None
    s = str(s).strip().replace(" ", "").replace(";", ":")
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2)):02d}"
    return None


def match_play_to_video_time(
    ocr_df: pd.DataFrame,
    period: int,
    clock_text: str,
    clock_sec: int,
    tolerance: float
):
    """
    1) try exact text match on MM:SS (best)
    2) fallback numeric match on rounded seconds
    """
    dfp = ocr_df[ocr_df["period_calc"] == period]
    if dfp.empty:
        return None

    # ---- pass 1: text match ----
    target_txt = normalize_mmss_text(clock_text)
    if target_txt:
        txt_norm = dfp["game_clock_text"].apply(normalize_mmss_text)
        hits = dfp[txt_norm == target_txt]
        if not hits.empty:
            row = hits.iloc[len(hits) // 2]  # stable pick
            sc = float(row["shot_clock_sec"]) if not pd.isna(
                row["shot_clock_sec"]) else None
            return float(row["video_time_sec"]), float(row["game_clock_sec"]), sc

    # ---- pass 2: numeric match ----
    arr = dfp["game_clock_sec"].to_numpy(dtype=float)
    arr_int = np.rint(arr)  # important: compare rounded seconds
    diffs = np.abs(arr_int - float(clock_sec))
    idx = int(np.argmin(diffs))
    if diffs[idx] > tolerance:
        return None

    row = dfp.iloc[idx]
    sc = float(row["shot_clock_sec"]) if not pd.isna(
        row["shot_clock_sec"]) else None
    return float(row["video_time_sec"]), float(row["game_clock_sec"]), sc


def find_possession_start(ocr_df: pd.DataFrame, period: int, shot_video_t: float, shot_clock_full: float, lookback_sec: float = 45.0):
    """
    Find the most recent time before shot where shot_clock was ~full (reset).
    We look back up to lookback_sec, searching for shot_clock_sec >= full - 0.5.
    """
    dfp = ocr_df[ocr_df["period_calc"] == period].copy()
    dfp = dfp[dfp["video_time_sec"] <= shot_video_t]
    if dfp.empty:
        return None

    dfp = dfp[dfp["video_time_sec"] >= (shot_video_t - lookback_sec)]
    if dfp.empty:
        return None

    sc = dfp["shot_clock_sec"].to_numpy(dtype=float)
    vt = dfp["video_time_sec"].to_numpy(dtype=float)

    for i in range(len(vt) - 1, -1, -1):
        if np.isnan(sc[i]):
            continue
        if sc[i] >= (shot_clock_full - 0.5):
            return float(vt[i])

    return None


# ----------------------------
# Cutting
# ----------------------------

def cut_clip(ffmpeg, video_path, out_path, start_t, end_t, reencode=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_t - start_t)

    if reencode:
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostats",
            "-y",
            "-ss", f"{start_t:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac",
            str(out_path)
        ]
    else:
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostats",
            "-y",
            "-ss", f"{start_t:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_path)
        ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_id", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--game_name", default=None)
    ap.add_argument("--metadata_root", default="data/metadata")
    ap.add_argument("--clips_root", default="data/3p_clips")
    ap.add_argument("--pre_sec", type=float, default=13.0)
    ap.add_argument("--post_sec", type=float, default=2.0)
    ap.add_argument("--shot_clock_full", type=float, default=30.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--no_reencode", action="store_true")
    ap.add_argument("--season_year", type=int, default=None,
                    help="Override season year (default: from pbp header)")
    ap.add_argument("--season_type", type=int, default=2,
                    help="ESPN season type (often 2 for regular season)")
    args = ap.parse_args()

    game_id = args.game_id
    video_path = Path(args.video)
    game_name = args.game_name or video_path.stem

    meta_dir = Path(args.metadata_root) / game_id
    pbp_path = meta_dir / "pbp.json"
    ocr_path = meta_dir / "ocr_map.csv"

    if not pbp_path.exists():
        raise FileNotFoundError(f"Missing pbp.json at: {pbp_path}")
    if not ocr_path.exists():
        raise FileNotFoundError(f"Missing ocr_map.csv at: {ocr_path}")

    pbp_json = load_json(pbp_path)

    season_year = args.season_year
    if season_year is None:
        season_year = pbp_json.get("header", {}).get("season", {}).get("year")
    if season_year is None:
        # safe fallback for modern NCAA games
        season_year = 2025

    plays3 = extract_three_pt_plays(pbp_json)
    print(
        f"Found {len(plays3)} 3PT plays in pbp. Season year={season_year} (type={args.season_type}).")

    ocr_df = pd.read_csv(ocr_path)
    ocr_df["video_time_sec"] = pd.to_numeric(
        ocr_df["video_time_sec"], errors="coerce")
    ocr_df["game_clock_sec"] = pd.to_numeric(
        ocr_df["game_clock_sec"], errors="coerce")
    ocr_df["shot_clock_sec"] = pd.to_numeric(
        ocr_df["shot_clock_sec"], errors="coerce")
    ocr_df["game_clock_text"] = ocr_df["game_clock_text"].astype(str)

    ocr_df = add_period_from_clock_jumps(ocr_df)
    print(
        f"✅ Periods detected: {ocr_df['period_calc'].min()} → {ocr_df['period_calc'].max()}"
    )

    # Output folder
    out_dir = Path(args.clips_root) / game_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Player stats cache (so reruns are fast)
    cache_path = meta_dir / "athlete_stats_cache.json"
    if cache_path.exists():
        try:
            athlete_cache = load_json(cache_path)
        except Exception:
            athlete_cache = {}
    else:
        athlete_cache = {}

    def get_cached_athlete(athlete_id: str):
        return athlete_cache.get(str(athlete_id))

    def set_cached_athlete(athlete_id: str, payload: dict):
        athlete_cache[str(athlete_id)] = payload

    # 1) First match everything (so we can sort by shot_t and name files in correct order)
    matched = []
    missed = 0

    for p in plays3:
        if p["period"] is None or p["clock_sec"] is None:
            missed += 1
            continue

        m = match_play_to_video_time(
            ocr_df,
            int(p["period"]),
            p["clock_text"],
            int(p["clock_sec"]),
            args.tolerance,
        )

        if m is None:
            missed += 1
            continue

        shot_t, matched_gc, matched_sc = m
        matched.append((p, shot_t, matched_gc, matched_sc))

    # Sort by actual video time (chronological)
    matched.sort(key=lambda x: x[1])

    # 2) Cut clips in sorted order + enrich metadata (shooter + season 3PT stats)
    meta_rows = []
    skipped_existing = 0
    cut_new = 0

    for clip_idx, (p, shot_t, matched_gc, matched_sc) in enumerate(
        tqdm(matched, desc="Generating clips", unit="clip"), start=1
    ):
        poss_start = find_possession_start(
            ocr_df,
            int(p["period"]),
            shot_t,
            args.shot_clock_full,
            lookback_sec=45.0,
        )

        clip_start = max(0.0, shot_t - args.pre_sec)
        if poss_start is not None:
            clip_start = max(clip_start, poss_start)
        clip_end = shot_t + args.post_sec

        # Fetch shooter stats (cached)
        shooter_id = p.get("shooter_id")
        shooter_name = p.get("shooter_name_guess")

        shooter_3pm = None
        shooter_3pa = None
        shooter_3ppct = None

        if shooter_id:
            cached = get_cached_athlete(shooter_id)
            if cached is None:
                name_api, m3, a3, pct3 = fetch_player_name_and_3pt_stats(
                    athlete_id=shooter_id,
                    season_year=int(season_year),
                    season_type=int(args.season_type),
                )
                cached = {
                    "athlete_id": shooter_id,
                    "name": name_api,
                    "season_year": int(season_year),
                    "season_type": int(args.season_type),
                    "3pm": m3,
                    "3pa": a3,
                    "3ppct": pct3,
                }
                set_cached_athlete(shooter_id, cached)

            shooter_name = cached.get("name") or shooter_name
            shooter_3pm = cached.get("3pm")
            shooter_3pa = cached.get("3pa")
            shooter_3ppct = cached.get("3ppct")

        # filename: stable ordered index + info
        made_tag = "MADE" if p["made"] else "MISS"
        clock_tag = (p["clock_text"] or "NA").replace(":", "_")
        team_tag = safe_name(p.get("team_name") or "TEAM")
        shooter_tag = safe_name(shooter_name or "UNKNOWN")

        # padded index guarantees dataset ordering
        fname = (
            f"{clip_idx:04d}_"
            f"{safe_name(game_name)}_"
            f"P{p['period']}_"
            f"{clock_tag}_"
            f"{team_tag}_"
            f"{shooter_tag}_"
            f"{made_tag}_"
            f"{p.get('sequence')}"
            f".mp4"
        )
        out_path = out_dir / fname

        if out_path.exists():
            skipped_existing += 1
        else:
            cut_clip(
                ffmpeg=args.ffmpeg,
                video_path=video_path,
                out_path=out_path,
                start_t=clip_start,
                end_t=clip_end,
                reencode=(not args.no_reencode),
            )
            cut_new += 1

        meta_rows.append({
            "clip_index": clip_idx,
            "clip_path": str(out_path),
            "game_id": game_id,
            "game_name": game_name,

            "play_id": p["play_id"],
            "sequence": p["sequence"],
            "period_pbp": p["period"],
            "pbp_clock_text": p["clock_text"],
            "pbp_clock_sec": p["clock_sec"],

            "matched_video_time_sec": shot_t,
            "matched_game_clock_sec": matched_gc,
            "matched_shot_clock_sec": matched_sc,
            "possession_start_video_time_sec": poss_start,

            "clip_start_sec": clip_start,
            "clip_end_sec": clip_end,

            "made": p["made"],
            "team": p.get("team_name"),
            "text": p.get("text", ""),

            "shooter_id": shooter_id,
            "shooter_name": shooter_name,
            "season_year": int(season_year),
            "season_type": int(args.season_type),
            "shooter_3pm": shooter_3pm,
            "shooter_3pa": shooter_3pa,
            "shooter_3ppct": shooter_3ppct,
        })

    # Save metadata CSV
    meta_csv = out_dir / "clips_metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(
            meta_rows[0].keys()) if meta_rows else ["clip_path"])
        w.writeheader()
        for r in meta_rows:
            w.writerow(r)

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(athlete_cache, f, indent=2)
    except Exception:
        pass

    print(f"\n✅ Saved {len(meta_rows)} clips to: {out_dir}")
    print(f"✅ Metadata: {meta_csv}")
    print(
        f"ℹ️  New clips cut: {cut_new} | Skipped existing: {skipped_existing}")
    if missed:
        print(
            f"⚠️  {missed} plays could not be matched (tolerance={args.tolerance}).")


if __name__ == "__main__":
    main()
