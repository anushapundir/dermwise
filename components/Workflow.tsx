import { Upload, Cpu, FileText } from "lucide-react";

/** Three‑step workflow cards explaining how DermWise works. */
const steps = [
  {
    icon: Upload,
    title: "Upload Image",
    description:
      "Drag & drop or browse for a dermoscopic JPG / PNG image of the skin lesion.",
  },
  {
    icon: Cpu,
    title: "AI Analyzes",
    description:
      "Our model classifies the lesion into one of the supported categories using deep learning.",
  },
  {
    icon: FileText,
    title: "Get a Report",
    description:
      "Receive a structured clinical report with classification, confidence, and guidance notes.",
  },
];

export default function Workflow() {
  return (
    <section
      id="how-it-works"
      className="scroll-mt-20 bg-slate-50 py-20 px-5"
    >
      <div className="mx-auto max-w-5xl text-center">
        {/* Section heading */}
        <h2 className="text-2xl font-bold text-slate-900 sm:text-3xl">
          How It Works
        </h2>
        <p className="mt-2 text-sm text-slate-500">
          Three simple steps from upload to clinical report.
        </p>

        {/* Cards */}
        <div className="mt-12 grid gap-6 sm:grid-cols-3">
          {steps.map((s, i) => (
            <div
              key={s.title}
              className="group relative rounded-2xl border border-slate-200 bg-white p-8 text-left shadow-sm transition-shadow hover:shadow-md"
            >
              {/* Step number */}
              <span className="absolute right-5 top-5 text-xs font-semibold text-slate-300">
                0{i + 1}
              </span>

              {/* Icon */}
              <div className="mb-5 grid h-11 w-11 place-items-center rounded-xl bg-brand-50 text-brand-600 transition-colors group-hover:bg-brand-100">
                <s.icon className="h-5 w-5" />
              </div>

              <h3 className="text-base font-semibold text-slate-800">
                {s.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-500">
                {s.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
