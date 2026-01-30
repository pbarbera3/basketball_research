import { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const repo = process.env.HF_REPO_ID;
    const token = process.env.HF_TOKEN;

    if (!repo) return new Response("Missing env HF_REPO_ID", { status: 500 });
    if (!token) return new Response("Missing env HF_TOKEN", { status: 500 });

    const { searchParams } = new URL(req.url);
    const path = searchParams.get("path");
    if (!path) return new Response("Missing ?path=", { status: 400 });

    const hfUrl = `https://huggingface.co/datasets/${repo}/resolve/main/${path}`;

    const r = await fetch(hfUrl, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    if (!r.ok) {
      const msg = await r.text().catch(() => "");
      return new Response(`HF fetch failed ${r.status}: ${msg}`, { status: 500 });
    }

    const text = await r.text();
    return new Response(text, {
      status: 200,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return new Response(`Server error: ${msg}`, { status: 500 });
  }
}
