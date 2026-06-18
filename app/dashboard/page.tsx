"use client";

import {
  useState,
  useCallback,
  useRef,
  type DragEvent,
  type ChangeEvent,
} from "react";
import { Upload, ImageIcon, Trash2, FileWarning, Loader2 } from "lucide-react";
import { analyzeImage, type AnalysisResult, type ReportModel } from "@/lib/api";
import ReportDisplay from "@/components/ReportDisplay";

/** Accepted MIME types */
const ACCEPTED = ["image/jpeg", "image/png"];
const MAX_SIZE_MB = 10;

export default function DashboardPage() {
  // ── File state ──
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Analysis state ──
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [warmingUp, setWarmingUp] = useState(false);
  // Which model generates the report: "qwen" (hosted, fast) or "tinyllama" (fine-tuned, local)
  const [model, setModel] = useState<ReportModel>("qwen");

  // ── Validation ──
  const validate = (f: File): string | null => {
    if (!ACCEPTED.includes(f.type))
      return "Only JPG and PNG images are accepted.";
    if (f.size > MAX_SIZE_MB * 1024 * 1024)
      return `File exceeds ${MAX_SIZE_MB} MB limit.`;
    return null;
  };

  // ── Handle selected file ──
  const handleFile = useCallback((f: File) => {
    const err = validate(f);
    if (err) {
      setError(err);
      setFile(null);
      setPreview(null);
      return;
    }
    setError(null);
    setResult(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  // ── Drag events ──
  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  };

  // ── Browse ──
  const onBrowse = () => inputRef.current?.click();
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
  };

  // ── Remove image & reset ──
  const resetAll = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setError(null);
    setResult(null);
    setLoading(false);
    setWarmingUp(false);
    if (inputRef.current) inputRef.current.value = "";
  };

  // ── Analyze image ──
  const handleAnalyze = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setLoading(true);
    setWarmingUp(false);

    // Show "warming up" if request takes > 8s (HF Space cold start)
    const warmupTimer = setTimeout(() => setWarmingUp(true), 8000);

    try {
      const data = await analyzeImage(file, model);
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Analysis failed. Please try again."
      );
    } finally {
      clearTimeout(warmupTimer);
      setLoading(false);
      setWarmingUp(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl px-5 py-14">
      {/* ── Page header ── */}
      <h1 className="text-2xl font-bold text-slate-900 sm:text-3xl">
        Dashboard
      </h1>
      <p className="mt-1 text-sm text-slate-500">
        Upload a dermoscopic skin lesion image for AI-powered analysis.
      </p>

      {/* ── Upload card ── */}
      <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <h2 className="text-lg font-semibold text-slate-800">
          Upload a lesion image
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Accepted formats: <strong>JPG, PNG</strong> &middot; Max{" "}
          {MAX_SIZE_MB} MB &middot; Use well‑lit, focused dermoscopic images for
          best results.
        </p>

        {/* ── Drop zone ── */}
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={!loading ? onBrowse : undefined}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && !loading && onBrowse()}
          aria-label="Upload area — click or drag and drop an image"
          className={`mt-5 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 transition-colors ${
            dragOver
              ? "border-brand-400 bg-brand-50"
              : "border-slate-200 bg-slate-50 hover:border-brand-300 hover:bg-brand-50/40"
          } ${loading ? "pointer-events-none opacity-60" : ""}`}
        >
          {preview ? (
            <div className="relative w-full max-w-xs">
              <img
                src={preview}
                alt="Selected lesion"
                className="w-full rounded-lg object-contain shadow"
              />
              {!loading && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    resetAll();
                  }}
                  className="absolute -right-2 -top-2 grid h-7 w-7 place-items-center rounded-full bg-red-100 text-red-600 shadow hover:bg-red-200 transition-colors"
                  aria-label="Remove image"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          ) : (
            <>
              <div className="grid h-12 w-12 place-items-center rounded-full bg-brand-100 text-brand-600">
                <Upload className="h-5 w-5" />
              </div>
              <p className="text-sm text-slate-500">
                Drag &amp; drop an image here, or{" "}
                <span className="font-semibold text-brand-600">
                  browse files
                </span>
              </p>
            </>
          )}
        </div>

        {/* Hidden file input */}
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          className="hidden"
          onChange={onChange}
          aria-hidden
        />

        {/* ── Error message ── */}
        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <FileWarning className="h-4 w-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* ── File info ── */}
        {file && !error && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-800">
            <ImageIcon className="h-4 w-4 flex-shrink-0" />
            <span className="truncate">{file.name}</span>
            <span className="ml-auto whitespace-nowrap text-xs text-brand-600">
              {(file.size / 1024).toFixed(0)} KB
            </span>
          </div>
        )}

        {/* ── Report model toggle ── */}
        {file && !error && !result && (
          <div className="mt-5">
            <p className="mb-2 text-xs font-medium text-slate-500">
              Report model
            </p>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setModel("qwen")}
                disabled={loading}
                className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-60 ${
                  model === "qwen"
                    ? "border-brand-500 bg-brand-50 text-brand-700"
                    : "border-slate-200 text-slate-600 hover:border-slate-300"
                }`}
              >
                Qwen‑7B <span className="text-xs font-normal text-slate-400">· fast</span>
              </button>
              <button
                type="button"
                onClick={() => setModel("tinyllama")}
                disabled={loading}
                className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-60 ${
                  model === "tinyllama"
                    ? "border-brand-500 bg-brand-50 text-brand-700"
                    : "border-slate-200 text-slate-600 hover:border-slate-300"
                }`}
              >
                TinyLlama <span className="text-xs font-normal text-slate-400">· fine‑tuned</span>
              </button>
            </div>
            {model === "tinyllama" && (
              <p className="mt-2 text-xs text-amber-600">
                Runs the fine‑tuned QLoRA model on free CPU — the first request can
                take 1–3 minutes.
              </p>
            )}
          </div>
        )}

        {/* ── Analyze button ── */}
        {file && !error && !result && (
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="mt-5 w-full rounded-lg bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                {warmingUp
                  ? "Model is warming up — this may take a minute…"
                  : "Analyzing…"}
              </>
            ) : (
              "Analyze Image"
            )}
          </button>
        )}
      </div>

      {/* ── Results or placeholder ── */}
      {result ? (
        <ReportDisplay result={result} onReset={resetAll} />
      ) : (
        !file && (
          <div className="mt-6 rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
            <div className="mx-auto mb-3 grid h-10 w-10 place-items-center rounded-full bg-slate-200 text-slate-400">
              <FileWarning className="h-5 w-5" />
            </div>
            <h3 className="text-sm font-semibold text-slate-500">
              Analysis Results
            </h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">
              Upload an image above to get an AI-powered classification and
              clinical report.
            </p>
            <div className="mx-auto mt-5 max-w-sm space-y-2.5">
              <div className="h-3 w-full rounded bg-slate-200" />
              <div className="h-3 w-5/6 rounded bg-slate-200" />
              <div className="h-3 w-4/6 rounded bg-slate-200" />
              <div className="h-3 w-3/6 rounded bg-slate-200" />
            </div>
          </div>
        )
      )}
    </div>
  );
}
