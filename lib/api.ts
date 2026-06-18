/**
 * API client for calling the DermWise inference backend.
 *
 * In production, calls go through our Next.js API route (/api/analyze)
 * which proxies to the HuggingFace Space. This keeps the HF URL
 * hidden from the browser and avoids CORS issues.
 */

// ── Types ──

export interface TopPrediction {
  className: string;
  probability: number;
}

/** Which model generates the clinical report. */
export type ReportModel = "qwen" | "tinyllama";

export interface AnalysisResult {
  /** The predicted lesion class (e.g. "Melanoma (MEL)") */
  predictedClass: string;
  /** Confidence score 0–1 */
  confidence: number;
  /** Top-3 predictions with probabilities */
  topK: TopPrediction[];
  /** AI-generated clinical report text */
  report: string;
  /** Retrieved knowledge context used for report generation */
  retrievedContext?: string;
  /** Which model actually produced the report (may differ on fallback) */
  modelUsed?: string;
}

export interface AnalysisError {
  error: string;
  detail?: string;
}

// ── API call ──

/**
 * Send an image to the backend for analysis.
 * Goes through /api/analyze (Next.js route handler) which proxies to HF Space.
 */
export async function analyzeImage(
  file: File,
  model: ReportModel = "qwen"
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", model);

  const res = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      (body as AnalysisError).error ||
        `Analysis failed (HTTP ${res.status})`
    );
  }

  return res.json();
}
