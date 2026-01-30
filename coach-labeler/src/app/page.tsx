"use client";

import { useEffect, useState } from "react";
import { supabase } from "../../lib/supabaseClient";

export default function Home() {
  const [error, setError] = useState<string | null>(null);

  // If already logged in, go straight to /label
  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (data.session) window.location.href = "/label";
    });
  }, []);

  async function signInGoogle() {
    setError(null);
    const origin = window.location.origin;

    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${origin}/auth/callback`,
      },
    });

    if (error) setError(error.message);
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-2xl border p-6 shadow-sm">
        <h1 className="text-2xl font-semibold">Coach Labeling</h1>
        <p className="text-sm text-gray-500 mt-1">Sign in with Google.</p>

        <button
          className="mt-5 w-full rounded-xl bg-black text-white p-3"
          onClick={signInGoogle}
        >
          Continue with Google
        </button>

        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </div>
    </main>
  );
}
