export const runtime = "nodejs";

type TreeItem = {
  path: string;
  type: "file" | "directory";
};

export async function GET() {
  const repo = process.env.HF_REPO_ID;
  const token = process.env.HF_TOKEN;
  if (!repo) return new Response("Missing env HF_REPO_ID", { status: 500 });
  if (!token) return new Response("Missing env HF_TOKEN", { status: 500 });

  const apiUrl = `https://huggingface.co/api/datasets/${repo}/tree/main?recursive=1`;

  const r = await fetch(apiUrl, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });

  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    return new Response(`HF tree failed ${r.status}: ${msg}`, { status: 500 });
  }

  const items = (await r.json()) as TreeItem[];

  // Find <GAME>/clips_metadata.csv
  const csvFiles = items.filter(
    (x) =>
      x.type === "file" &&
      x.path.toLowerCase().endsWith("/clips_metadata.csv")
  );

  const games = csvFiles
    .map((f) => {
      const parts = f.path.split("/");
      const game_name = parts[0]; // folder at root
      return {
        game_id: game_name,        // use folder name as stable id
        game_name,
        csv_path: f.path,          // "<folder>/clips_metadata.csv"
      };
    })
    .sort((a, b) => a.game_name.localeCompare(b.game_name));

  return new Response(JSON.stringify({ games }), {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
