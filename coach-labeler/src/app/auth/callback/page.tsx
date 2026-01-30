"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "../../../../lib/supabaseClient";

export default function AuthCallback() {
  const router = useRouter();
  const [msg, setMsg] = useState("Signing you in...");

  useEffect(() => {
    (async () => {
      try {
        const url = new URL(window.location.href);
        const code = url.searchParams.get("code");

        // OAuth (Google) returns ?code=...
        if (code) {
          const { error } = await supabase.auth.exchangeCodeForSession(code);
          if (error) throw error;
        }

        // Small wait to ensure session is persisted
        for (let i = 0; i < 10; i++) {
          const { data } = await supabase.auth.getSession();
          if (data.session) {
            router.replace("/label");
            return;
          }
          await new Promise((r) => setTimeout(r, 150));
        }

        setMsg("Session not found. Please try signing in again.");
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : "Unknown error";
        setMsg(`Login failed: ${message}`);
      }
    })();
  }, [router]);

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="rounded-2xl border p-6">{msg}</div>
    </main>
  );
}
