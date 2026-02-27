"use client";

import Link from "next/link";
import { ArrowRight, ChevronDown } from "lucide-react";

/**
 * Hero section — full‑viewport height, heading + two CTAs.
 */
export default function Hero() {
  const scrollToWorkflow = () => {
    document
      .querySelector("#how-it-works")
      ?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center px-5 text-center">
      {/* Subtle gradient backdrop */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-b from-brand-50/60 via-white to-white"
      />

      {/* Badge */}
      <span className="mb-5 inline-block rounded-full border border-brand-200 bg-brand-50 px-4 py-1 text-xs font-semibold text-brand-700 tracking-wide">
        AI‑Powered Dermatology Assistant
      </span>

      {/* Heading */}
      <h1 className="max-w-2xl text-4xl font-bold leading-tight text-slate-900 sm:text-5xl lg:text-6xl">
        Skin Lesion Analysis
        <br />
        <span className="text-brand-600">&amp; Clinical Reporting</span>
      </h1>

      {/* Subtitle — 2 lines max */}
      <p className="mt-5 max-w-lg text-base leading-relaxed text-slate-500 sm:text-lg">
        Upload a dermoscopic image, get an AI classification and a structured
        clinical report — in seconds.
      </p>

      {/* CTAs */}
      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-md hover:bg-brand-700 transition-colors focus-visible:ring-2 focus-visible:ring-brand-400"
        >
          Try the Dashboard
          <ArrowRight className="h-4 w-4" />
        </Link>
        <button
          onClick={scrollToWorkflow}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-6 py-3 text-sm font-medium text-slate-600 hover:border-slate-300 hover:text-slate-900 transition-colors"
        >
          How it works
          <ChevronDown className="h-4 w-4" />
        </button>
      </div>
    </section>
  );
}
