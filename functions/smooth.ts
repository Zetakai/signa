// Cloudflare Pages Function: POST /smooth
// Converts ASL gloss (raw token transcript) into fluent English using Workers AI.
//
// Requires a Workers AI binding named `AI` (Pages -> Settings -> Functions ->
// Bindings -> add "Workers AI", variable name AI).

interface Env {
  AI: {
    run: (
      model: string,
      input: Record<string, unknown>
    ) => Promise<{ response?: string }>;
  };
}

const MODEL = '@cf/meta/llama-3.1-8b-instruct';

const SYSTEM_PROMPT =
  'You are a sign language interpreter. You receive ASL gloss: a sequence of ' +
  'uppercase signed words and fingerspelled words, in signing order, without ' +
  'grammar. Rewrite it as one natural, grammatically correct English sentence. ' +
  'Preserve the meaning. Do not add information. Reply with ONLY the sentence.';

export const onRequestPost: (ctx: {
  request: Request;
  env: Env;
}) => Promise<Response> = async ({ request, env }) => {
  try {
    const { gloss } = (await request.json()) as { gloss?: string };
    if (!gloss || !gloss.trim()) {
      return json({ text: '' });
    }

    const result = await env.AI.run(MODEL, {
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: `ASL gloss: ${gloss}\nEnglish:` },
      ],
      max_tokens: 128,
    });

    const text = (result.response ?? '').trim();
    return json({ text: text || gloss });
  } catch (err) {
    return json({ text: '', error: (err as Error).message }, 500);
  }
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
