import { AlertTriangle } from "lucide-react";

/**
 * Supported lesion classes — easy to update later when
 * the real model class list is finalized.
 */
const SUPPORTED_CLASSES = [
  "Melanocytic Nevi (NV)",
  "Melanoma (MEL)",
  "Benign Keratosis (BKL)",
  "Basal Cell Carcinoma (BCC)",
  "Actinic Keratosis (AKIEC)",
  "Vascular Lesions (VASC)",
  "Dermatofibroma (DF)",
];

/** Disclaimer callout — visually distinct bordered card. */
export default function Disclaimer() {
  return (
    <section id="disclaimer" className="scroll-mt-20 py-20 px-5">
      <div className="mx-auto max-w-3xl rounded-2xl border border-amber-200 bg-amber-50/50 p-8 sm:p-10">
        {/* Header */}
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-500" />
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Important Disclaimer
            </h2>
            <p className="mt-1 text-sm leading-relaxed text-slate-600">
              DermAssist is an <strong>educational &amp; research tool</strong>.
              It is <em>not</em> intended as a substitute for professional
              medical advice, diagnosis, or treatment. Always consult a
              qualified dermatologist for clinical decisions.
            </p>
          </div>
        </div>

        {/* Supported classes */}
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-slate-700">
            Supported Lesion Classes
          </h3>
          <ul className="mt-2 grid gap-x-6 gap-y-1 text-sm text-slate-600 sm:grid-cols-2">
            {SUPPORTED_CLASSES.map((c) => (
              <li key={c} className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-brand-400" />
                {c}
              </li>
            ))}
          </ul>
        </div>

        {/* Additional notes */}
        <p className="mt-6 text-xs leading-relaxed text-slate-400">
          The classification model was trained on the HAM10000 dataset and may
          not generalize to all clinical scenarios. Model accuracy should be
          validated independently before any clinical reliance.
        </p>
      </div>
    </section>
  );
}
