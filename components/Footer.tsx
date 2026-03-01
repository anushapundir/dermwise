import Link from "next/link";
import { Stethoscope } from "lucide-react";

/** Minimal site‑wide footer. */
export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-slate-100 bg-slate-50">
      <div className="mx-auto flex max-w-6xl flex-col items-center gap-4 px-5 py-8 sm:flex-row sm:justify-between">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <Stethoscope className="h-4 w-4 text-brand-600" />
          <span className="text-sm font-semibold text-slate-700">
            DermWise
          </span>
        </div>

        {/* Links */}
        <div className="flex items-center gap-5 text-xs text-slate-400">
          <Link href="/" className="hover:text-slate-600 transition-colors">
            Home
          </Link>
          <Link
            href="/dashboard"
            className="hover:text-slate-600 transition-colors"
          >
            Dashboard
          </Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-600 transition-colors"
          >
            GitHub
          </a>
        </div>

        {/* Copyright */}
        <p className="text-xs text-slate-400">
          &copy; {year} DermWise. For research &amp; education only.
        </p>
      </div>
    </footer>
  );
}
