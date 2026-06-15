import { NextRequest, NextResponse } from "next/server";

/**
 * POST /api/analyze
 *
 * Server-side proxy that forwards the uploaded image to the
 * HuggingFace Space Gradio API. This keeps the HF URL secret
 * and avoids browser CORS issues.
 *
 * Gradio 5.x (with SSR) uses a two-step API:
 *   1. POST /gradio_api/call/analyze  →  { event_id }
 *   2. GET  /gradio_api/call/analyze/{event_id}  →  SSE stream with result
 */

const HF_SPACE_URL = process.env.HF_SPACE_URL || "";

export async function POST(req: NextRequest) {
  // ── Validate env ──
  if (!HF_SPACE_URL) {
    return NextResponse.json(
      { error: "Backend not configured. Set HF_SPACE_URL env variable." },
      { status: 503 }
    );
  }

  try {
    // ── Read the uploaded file from formData ──
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json(
        { error: "No file uploaded." },
        { status: 400 }
      );
    }

    // ── Convert file to base64 for Gradio API ──
    const bytes = await file.arrayBuffer();
    const base64 = Buffer.from(bytes).toString("base64");
    const mimeType = file.type || "image/jpeg";
    const dataUri = `data:${mimeType};base64,${base64}`;

    // ── Step 1: Initiate the prediction ──
    // Gradio 5.x uses the function name as the endpoint (/analyze)
    // Image input requires {url: dataUri} format
    const callRes = await fetch(`${HF_SPACE_URL}/gradio_api/call/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [{ url: dataUri }] }),
    });

    if (!callRes.ok) {
      const detail = await callRes.text().catch(() => "Unknown error");
      console.error("[/api/analyze] HF call error:", callRes.status, detail);
      return NextResponse.json(
        { error: "AI backend returned an error.", detail },
        { status: 502 }
      );
    }

    const { event_id } = await callRes.json();
    if (!event_id) {
      return NextResponse.json(
        { error: "AI backend did not return an event ID." },
        { status: 502 }
      );
    }

    // ── Step 2: Poll for the result via SSE stream ──
    // The HF Space may cold-start (free tier sleeps when idle) and CPU-side
    // classification + the remote LLM call take time. Allow up to 5 minutes.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300000);

    const resultRes = await fetch(
      `${HF_SPACE_URL}/gradio_api/call/analyze/${event_id}`,
      { method: "GET", signal: controller.signal }
    );

    clearTimeout(timeout);

    if (!resultRes.ok) {
      const detail = await resultRes.text().catch(() => "Unknown error");
      console.error("[/api/analyze] HF result error:", resultRes.status, detail);
      return NextResponse.json(
        { error: "AI backend failed to return results.", detail },
        { status: 502 }
      );
    }

    // Parse SSE response — look for the "complete" event with data
    const sseText = await resultRes.text();
    const lines = sseText.split("\n");

    let resultData: unknown = null;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].startsWith("event: complete")) {
        // The next "data:" line contains the JSON payload
        const dataLine = lines[i + 1];
        if (dataLine && dataLine.startsWith("data: ")) {
          resultData = JSON.parse(dataLine.slice(6));
        }
        break;
      }
      if (lines[i].startsWith("event: error")) {
        const dataLine = lines[i + 1];
        const errMsg = dataLine?.startsWith("data: ") ? dataLine.slice(6) : "Unknown error";
        console.error("[/api/analyze] HF event error:", errMsg);
        return NextResponse.json(
          { error: "AI analysis failed.", detail: errMsg },
          { status: 502 }
        );
      }
    }

    if (!resultData) {
      console.error("[/api/analyze] No result data in SSE:", sseText.slice(0, 500));
      return NextResponse.json(
        { error: "AI backend returned empty results." },
        { status: 502 }
      );
    }

    // Gradio SSE returns the data array directly: [{...}]
    // Extract the first element (our single JSON output)
    const dataArray = Array.isArray(resultData)
      ? resultData
      : (resultData as { data?: unknown[] }).data || [];
    const resultRaw = dataArray[0];

    if (!resultRaw) {
      console.error("[/api/analyze] Empty data array:", JSON.stringify(resultData).slice(0, 500));
      return NextResponse.json(
        { error: "AI backend returned empty results." },
        { status: 502 }
      );
    }

    const result =
      typeof resultRaw === "string" ? JSON.parse(resultRaw) : resultRaw;

    // ── Map to our frontend format ──
    return NextResponse.json({
      predictedClass: result.predicted_class,
      confidence: result.confidence,
      topK: (result.top_k || []).map(
        (item: { class: string; prob: number }) => ({
          className: item.class,
          probability: item.prob,
        })
      ),
      report: result.report || "",
      retrievedContext: result.retrieved_context || "",
    });
  } catch (err) {
    console.error("[/api/analyze] Error:", err);
    return NextResponse.json(
      {
        error: "Failed to process the image. Please try again.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}
