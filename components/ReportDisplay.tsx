import type { AnalysisResult } from "@/lib/api";
import { CheckCircle, AlertTriangle, BarChart3, FileText, RotateCcw } from "lucide-react";

// ── Class metadata for severity badges ──
const SEVERITY_MAP: Record<string, { label: string; color: string }> = {
  akiec: { label: "Pre-cancerous", color: "bg-amber-100 text-amber-800" },
  bcc:   { label: "Malignant",     color: "bg-red-100 text-red-800" },
  bkl:   { label: "Benign",        color: "bg-green-100 text-green-800" },
  df:    { label: "Benign",        color: "bg-green-100 text-green-800" },
  mel:   { label: "Malignant",     color: "bg-red-100 text-red-800" },
  nv:    { label: "Benign",        color: "bg-green-100 text-green-800" },
  vasc:  { label: "Benign",        color: "bg-green-100 text-green-800" },
};

/** Extract the short class code from a class name like "Melanoma (MEL)" → "mel" */
function getClassCode(className: string): string {
  const match = className.match(/\((\w+)\)/);
  return match ? match[1].toLowerCase() : className.toLowerCase();
}

interface Props {
  result: AnalysisResult;
  onReset: () => void;
}

/**
 * Displays the AI analysis results:
 * - Classification badge with severity
 * - Confidence bar
 * - Top-3 predictions
 * - Clinical report text
 */
export default function ReportDisplay({ result, onReset }: Props) {
  const code = getClassCode(result.predictedClass);
  const severity = SEVERITY_MAP[code] ?? { label: "Unknown", color: "bg-slate-100 text-slate-600" };

  return (
    <div className="mt-6 space-y-5">
      {/* ── Classification result ── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Classification Result
            </p>
            <h3 className="mt-1 text-xl font-bold text-slate-900">
              {result.predictedClass}
            </h3>
            <span
              className={`mt-2 inline-block rounded-full px-3 py-0.5 text-xs font-semibold ${severity.color}`}
            >
              {severity.label}
            </span>
          </div>
          <div className="text-right">
            <p className="text-xs font-medium text-slate-400">Confidence</p>
            <p className="mt-1 text-2xl font-bold text-brand-600">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-4">
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-brand-500 transition-all duration-500"
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* ── Top-3 predictions ── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div className="mb-4 flex items-center gap-2 text-slate-700">
          <BarChart3 className="h-4 w-4" />
          <h4 className="text-sm font-semibold">Top Predictions</h4>
        </div>
        <div className="space-y-3">
          {result.topK.map((pred, i) => {
            const pCode = getClassCode(pred.className);
            const pSev = SEVERITY_MAP[pCode];
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="w-5 text-center text-xs font-semibold text-slate-400">
                  {i + 1}
                </span>
                <div className="flex-1">
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm font-medium text-slate-700">
                      {pred.className}
                    </span>
                    <span className="text-xs font-semibold text-slate-500">
                      {(pred.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        i === 0 ? "bg-brand-500" : "bg-brand-300"
                      }`}
                      style={{ width: `${pred.probability * 100}%` }}
                    />
                  </div>
                </div>
                {pSev && (
                  <span
                    className={`hidden rounded-full px-2 py-0.5 text-[10px] font-semibold sm:inline-block ${pSev.color}`}
                  >
                    {pSev.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Clinical report ── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div className="mb-4 flex items-center gap-2 text-slate-700">
          <FileText className="h-4 w-4" />
          <h4 className="text-sm font-semibold">Clinical Report</h4>
        </div>
        <div className="prose prose-sm max-w-none text-slate-600">
          {result.report.split("\n").map((line, i) =>
            line.trim() ? (
              <p key={i} className="mb-2 leading-relaxed">
                {line}
              </p>
            ) : null
          )}
        </div>
      </div>

      {/* ── Disclaimer + Reset ── */}
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
        <p className="flex items-center gap-1.5 text-xs text-slate-400">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          For educational use only — consult a dermatologist for clinical
          decisions.
        </p>
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:border-slate-300 hover:text-slate-900 transition-colors"
        >
          <RotateCcw className="h-4 w-4" />
          Analyze Another
        </button>
      </div>
    </div>
  );
}
