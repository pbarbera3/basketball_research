"use client";

import { useEffect, useMemo, useState } from "react";
import { supabase } from "../../../lib/supabaseClient";
import Papa from "papaparse";

type GameRow = {
  game_id: string;
  game_name: string;
  csv_path: string;
  num_clips?: string;
};

type ClipRow = {
  clip_index: string;
  clip_path: string;
  game_id: string;
  game_name: string;
  play_id: string;
  made: string;
  team: string;
  text: string;
  shooter_id: string;
  shooter_name: string;
  shooter_3pm: string;
  shooter_3pa: string;
  shooter_3ppct: string;
};

type ExistingLabel = {
  label: "good" | "bad" | "skip";
  skip_reason: "bad_cut" | "neutral" | null;
} | null;

function toHfVideoPath(clip_path: string) {
  const p = clip_path.replace(/\\/g, "/");
  return p.startsWith("data/3p_clips/") ? p.replace(/^data\/3p_clips\//, "") : p;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

export default function LabelPage() {
  const [userId, setUserId] = useState<string | null>(null);

  const [games, setGames] = useState<GameRow[]>([]);
  const [gameIndex, setGameIndex] = useState(0);

  const [clips, setClips] = useState<ClipRow[]>([]);
  const [clipIndex, setClipIndex] = useState(0);

  const [loadingClips, setLoadingClips] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [progressIndex, setProgressIndex] = useState(0);
  const [showSkipReasons, setShowSkipReasons] = useState(false);

  const [existingLabel, setExistingLabel] = useState<ExistingLabel>(null);
  const [toast, setToast] = useState<string | null>(null);

  const currentGame = games[gameIndex];
  const clip = clips[clipIndex];

  const clipUrl = clip?.clip_path
    ? `/api/hf-video?path=${encodeURIComponent(toHfVideoPath(clip.clip_path))}`
    : null;

  const shooter = clip?.shooter_name ?? "";
  const makes = clip?.shooter_3pm ? Number(clip.shooter_3pm) : null;
  const atts = clip?.shooter_3pa ? Number(clip.shooter_3pa) : null;
  const pct = clip?.shooter_3ppct ? Number(clip.shooter_3ppct) : null;

  const clipCount = clips.length || 0;
  const progressPct = clipCount ? Math.round(((clipIndex + 1) / clipCount) * 100) : 0;

  // ---------- Auth gate ----------
  useEffect(() => {
    let cancelled = false;

    async function init() {
      for (let i = 0; i < 10; i++) {
        const { data } = await supabase.auth.getSession();
        const session = data.session;
        if (session) {
          if (!cancelled) setUserId(session.user.id);
          return;
        }
        await new Promise((r) => setTimeout(r, 200));
      }
      window.location.href = "/";
    }

    init();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      const id = session?.user.id ?? null;
      setUserId(id);
      if (!id) window.location.href = "/";
    });

    return () => {
      cancelled = true;
      listener.subscription.unsubscribe();
    };
  }, []);

  // ---------- Load games ----------
  useEffect(() => {
    if (!userId) return;

    async function loadManifest() {
      setErr(null);
      const res = await fetch("/api/hf-manifest", { cache: "no-store" });
      if (!res.ok) {
        const t = await res.text().catch(() => "");
        setErr(`Failed to load manifest: ${t}`);
        return;
      }
      const json = await res.json();
      setGames(json.games);
    }

    loadManifest();
  }, [userId]);

  // ---------- Load clips + progress for current game ----------
  useEffect(() => {
    if (!userId || !currentGame) return;

    async function loadGameClipsAndProgress() {
      setLoadingClips(true);
      setErr(null);
      setShowSkipReasons(false);
      setExistingLabel(null);

      try {
        // 1) CSV
        const csvUrl = `/api/hf-text?path=${encodeURIComponent(currentGame.csv_path)}`;
        const res = await fetch(csvUrl, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load game CSV (${res.status})`);

        const text = await res.text();
        const parsed = Papa.parse<ClipRow>(text, { header: true, skipEmptyLines: true });
        const rows = parsed.data.filter((r) => r.play_id && r.clip_path);

        setClips(rows);

        // 2) Progress
        const { data: progRow, error: progErr } = await supabase
          .from("coach_progress")
          .select("current_clip_index")
          .eq("user_id", userId)
          .eq("game_id", currentGame.game_id)
          .maybeSingle();

        if (progErr) throw progErr;

        const startIndex = progRow?.current_clip_index ?? 0;

        if (!progRow) {
          const { error: insErr } = await supabase.from("coach_progress").insert({
            user_id: userId,
            game_id: currentGame.game_id,
            current_clip_index: 0,
          });
          if (insErr) throw insErr;
        }

        setProgressIndex(startIndex);
        setClipIndex(clamp(startIndex, 0, Math.max(0, rows.length - 1)));
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setErr(msg);
      } finally {
        setLoadingClips(false);
      }
    }

    loadGameClipsAndProgress();
  }, [userId, currentGame?.game_id]);

  // ---------- Load existing label for this clip (for confidence + relabeling) ----------
  useEffect(() => {
    if (!userId || !currentGame || !clip) return;

    (async () => {
      const { data, error } = await supabase
        .from("labels")
        .select("label, skip_reason")
        .eq("user_id", userId)
        .eq("game_id", currentGame.game_id)
        .eq("clip_id", clip.play_id)
        .maybeSingle();

      if (error) return; // don’t block UX
      setExistingLabel((data as ExistingLabel) ?? null);
    })();
  }, [userId, currentGame?.game_id, clip?.play_id]);

  // ---------- Toast auto-hide ----------
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 1200);
    return () => clearTimeout(t);
  }, [toast]);

  if (!userId) {
    return (
      <main className="min-h-screen flex items-center justify-center p-6">
        <div className="rounded-2xl border p-6">Loading session...</div>
      </main>
    );
  }

  async function saveLabel(label: "good" | "bad" | "skip", skip_reason?: "bad_cut" | "neutral") {
    if (!clip || !currentGame) return;
    if (saving) return;

    setSaving(true);
    setErr(null);

    try {
      const clip_id = clip.play_id;
      const game_id = currentGame.game_id;

      const { error: labErr } = await supabase.from("labels").upsert(
        {
          user_id: userId,
          game_id,
          clip_id,
          label,
          skip_reason: label === "skip" ? skip_reason ?? null : null,
        },
        { onConflict: "user_id,game_id,clip_id" }
      );

      if (labErr) throw labErr;

      // Update UI label immediately
      setExistingLabel({
        label,
        skip_reason: label === "skip" ? (skip_reason ?? null) : null,
      });

      // Advance
      const nextIndex = clamp(clipIndex + 1, 0, Math.max(0, clips.length - 1));
      const newProgress = Math.max(progressIndex, nextIndex);

      const { error: progErr } = await supabase
        .from("coach_progress")
        .update({ current_clip_index: newProgress, updated_at: new Date().toISOString() })
        .eq("user_id", userId)
        .eq("game_id", game_id);

      if (progErr) throw progErr;

      setProgressIndex(newProgress);
      setClipIndex(nextIndex);
      setShowSkipReasons(false);
      setToast("Saved ✅");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setErr(msg);
    } finally {
      setSaving(false);
    }
  }

  function undoOne() {
    setShowSkipReasons(false);
    setClipIndex((i) => clamp(i - 1, 0, Math.max(0, clips.length - 1)));
    setToast("Back ↩️");
  }

  return (
    <main className="min-h-screen bg-black text-white">
      {/* Sticky Header */}
      <header className="sticky top-0 z-20 bg-black/90 backdrop-blur border-b border-white/10">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-xs text-white/60">
                Game {games.length ? gameIndex + 1 : 0}/{games.length || 0}
              </div>
              <div className="truncate text-sm font-semibold">
                {currentGame ? currentGame.game_name : "Loading game..."}
              </div>
            </div>

            <button
              className="shrink-0 rounded-xl border border-white/20 px-3 py-2 text-sm"
              onClick={async () => {
                await supabase.auth.signOut();
                window.location.href = "/";
              }}
            >
              Sign out
            </button>
          </div>

          <div className="mt-2 flex items-end justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">
                Clip {clipCount ? clipIndex + 1 : 0}/{clipCount}
              </div>
              <div className="text-xs text-white/60">
                {clip ? "Ready to label" : "Loading clip..."}
              </div>
            </div>

            {existingLabel && (
              <div className="text-xs rounded-full border border-white/15 px-3 py-1 text-white/80">
                Your label:{" "}
                <span className="font-semibold">
                  {existingLabel.label}
                  {existingLabel.label === "skip" && existingLabel.skip_reason
                    ? ` (${existingLabel.skip_reason})`
                    : ""}
                </span>
              </div>
            )}
          </div>

          {/* Progress bar */}
          <div className="mt-3 h-2 w-full rounded-full bg-white/10 overflow-hidden">
            <div className="h-full bg-white/80" style={{ width: `${progressPct}%` }} />
          </div>

          {/* De-emphasized game navigation (optional) */}
          <div className="mt-3 flex gap-2">
            <button
              className="rounded-xl border border-white/15 px-3 py-2 text-xs disabled:opacity-40"
              disabled={gameIndex === 0}
              onClick={() => setGameIndex((i) => Math.max(0, i - 1))}
            >
              Prev game
            </button>
            <button
              className="rounded-xl border border-white/15 px-3 py-2 text-xs disabled:opacity-40"
              disabled={gameIndex >= games.length - 1}
              onClick={() => setGameIndex((i) => Math.min(games.length - 1, i + 1))}
            >
              Next game
            </button>
          </div>
        </div>
      </header>

      {/* Body */}
      <div className="mx-auto max-w-3xl px-4 py-4 pb-28">
        {err && (
          <div className="rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-red-200">
            {err}
          </div>
        )}

        <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
          {loadingClips ? (
            <div className="text-white/60 p-6">Loading game clips...</div>
          ) : !clipUrl ? (
            <div className="text-white/60 p-6">No clip loaded.</div>
          ) : (
            <>
              <video
                key={clipUrl}
                src={clipUrl}
                controls
                autoPlay
                playsInline
                className="w-full rounded-xl bg-black"
              />

              {clip && (
                <div className="mt-3 rounded-2xl border border-white/10 bg-white/5 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs text-white/60">{clip.team}</div>

                      <div className="truncate text-base font-semibold text-white">
                        {clip.shooter_name || "Unknown shooter"}
                      </div>

                      <div className="mt-1 text-sm text-white/70">
                        {clip.made === "True" ? (
                          <span className="text-green-300 font-semibold">MADE</span>
                        ) : (
                          <span className="text-red-300 font-semibold">MISS</span>
                        )}
                        {" • "}
                        {makes !== null && atts !== null ? `${makes}/${atts}` : "—"}
                        {" • "}
                        {pct !== null ? `${pct.toFixed(1)}%` : "—"}
                      </div>
                    </div>

                    <div className="shrink-0 rounded-full border border-white/15 px-3 py-1 text-xs text-white/70">
                      #{clipIndex + 1}/{clips.length}
                    </div>
                  </div>
                </div>
              )}

            </>
          )}
        </div>
      </div>

      {/* Sticky Bottom Action Bar */}
      <footer className="fixed bottom-0 left-0 right-0 z-30 bg-black/92 backdrop-blur border-t border-white/10">
        <div className="mx-auto max-w-3xl px-4 py-3">
          {/* Skip reasons (inline) */}
          {showSkipReasons && (
            <div className="mb-3 rounded-2xl border border-white/10 bg-white/5 p-3">
              <div className="text-xs text-white/70 mb-2">Why skip?</div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  className="rounded-xl border border-white/15 px-3 py-3 text-sm disabled:opacity-40"
                  disabled={saving}
                  onClick={() => saveLabel("skip", "bad_cut")}
                >
                  Bad cut / can’t see
                </button>
                <button
                  className="rounded-xl border border-white/15 px-3 py-3 text-sm disabled:opacity-40"
                  disabled={saving}
                  onClick={() => saveLabel("skip", "neutral")}
                >
                  Neutral / too hard
                </button>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <button
              className="
                rounded-2xl bg-white text-black py-4 text-lg font-semibold
                transition-transform duration-75 active:scale-[0.98] active:brightness-95
                focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40
                disabled:opacity-50
              "
              disabled={saving || !clip}
              onClick={() => saveLabel("good")}
            >
              ✅ Good
            </button>

            <button
              className="
                rounded-2xl border border-white/25 py-4 text-lg font-semibold
                transition-transform duration-75 active:scale-[0.98] active:bg-white/5
                focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30
                disabled:opacity-50
              "
              disabled={saving || !clip}
              onClick={() => saveLabel("bad")}
            >
              ❌ Bad
            </button>
          </div>

          <div className="mt-2 flex items-center justify-between gap-2">
            <button
              className="rounded-xl border border-white/15 px-4 py-2 text-sm disabled:opacity-40"
              disabled={saving || !clip}
              onClick={() => setShowSkipReasons((v) => !v)}
            >
              {showSkipReasons ? "Close skip" : "Skip"}
            </button>

            <button
              className="rounded-xl border border-white/15 px-4 py-2 text-sm disabled:opacity-40"
              disabled={saving || clipIndex === 0}
              onClick={undoOne}
            >
              Undo
            </button>

            <div className="text-xs text-white/50">
              {saving ? "Saving..." : "Fast label mode"}
            </div>
          </div>

          {toast && (
            <div className="mt-2 text-center text-xs text-white/70">{toast}</div>
          )}
        </div>
      </footer>
    </main>
  );
}
