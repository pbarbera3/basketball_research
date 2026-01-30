import { NextRequest } from "next/server";

export const runtime = "nodejs";

function mustEnv(name: string) {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

export async function GET(req: NextRequest) {
  const repo = mustEnv("HF_REPO_ID");
  const token = mustEnv("HF_TOKEN");

  const { searchParams } = new URL(req.url);
  const path = searchParams.get("path");
  if (!path) return new Response("Missing ?path=", { status: 400 });

  const hfUrl = `https://huggingface.co/datasets/${repo}/resolve/main/${path}`;

  // Forward Range header so <video> seeking works
  const range = req.headers.get("range") ?? undefined;

  const r = await fetch(hfUrl, {
    headers: {
      Authorization: `Bearer ${token}`,
      ...(range ? { Range: range } : {}),
    },
    cache: "no-store",
  });

  if (!r.ok && r.status !== 206) {
    const msg = await r.text().catch(() => "");
    return new Response(`HF video fetch failed (${r.status}): ${msg}`, { status: 500 });
  }

  // Stream the body back to the browser
  // Stream the body back to the browser
  const headers = new Headers();

  // Force mp4 MIME if missing (prevents Safari/Chrome "unsupported format" error)
  headers.set("Content-Type", r.headers.get("content-type") ?? "video/mp4");

  const cl = r.headers.get("content-length");
  if (cl) headers.set("Content-Length", cl);

  headers.set("Accept-Ranges", r.headers.get("accept-ranges") ?? "bytes");

  const cr = r.headers.get("content-range");
  if (cr) headers.set("Content-Range", cr);

  headers.set("Cache-Control", "no-store");


  return new Response(r.body, {
    status: r.status, // 200 or 206
    headers,
  });
}
