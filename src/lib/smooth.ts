// Client for the optional LLM "smooth" Pages Function. Turns raw ASL gloss
// into fluent English. Gracefully no-ops when offline.

export interface SmoothResult {
  text: string;
  source: 'llm' | 'offline-fallback';
}

export function isOnline(): boolean {
  return typeof navigator === 'undefined' ? true : navigator.onLine;
}

export async function smoothTranscript(gloss: string): Promise<SmoothResult> {
  if (!gloss.trim()) return { text: '', source: 'offline-fallback' };
  if (!isOnline()) {
    return { text: gloss, source: 'offline-fallback' };
  }
  try {
    const res = await fetch('/smooth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gloss }),
    });
    if (!res.ok) throw new Error(`smooth failed: ${res.status}`);
    const data = (await res.json()) as { text?: string };
    return { text: data.text?.trim() || gloss, source: 'llm' };
  } catch {
    // Function not deployed / offline / error -> fall back to raw gloss.
    return { text: gloss, source: 'offline-fallback' };
  }
}
