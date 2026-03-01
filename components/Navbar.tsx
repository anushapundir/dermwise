"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, Stethoscope } from "lucide-react";

/** Fixed top navigation bar with mobile hamburger menu. */
export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const links = [
    { href: "/", label: "Home" },
    { href: "/#how-it-works", label: "How it works", scroll: true },
    { href: "/#disclaimer", label: "Disclaimer", scroll: true },
  ];

  /** Scroll‑to handler for same‑page anchors */
  const handleScroll = (hash: string) => {
    setOpen(false);
    const el = document.querySelector(hash);
    el?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav className="fixed inset-x-0 top-0 z-50 h-16 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
      <div className="mx-auto flex h-full max-w-6xl items-center justify-between px-5">
        {/* ── Logo ── */}
        <Link href="/" className="flex items-center gap-2 group">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 transition-colors group-hover:bg-brand-700">
            <Stethoscope className="h-4 w-4 text-white" />
          </div>
          <span className="text-lg font-bold text-slate-900 tracking-tight">
            DermWise
          </span>
        </Link>

        {/* ── Desktop links ── */}
        <div className="hidden md:flex items-center gap-6">
          {links.map((l) =>
            l.scroll ? (
              <button
                key={l.label}
                onClick={() =>
                  pathname === "/"
                    ? handleScroll(l.href.replace("/", ""))
                    : (window.location.href = l.href)
                }
                className="text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors"
              >
                {l.label}
              </button>
            ) : (
              <Link
                key={l.label}
                href={l.href}
                className={`text-sm font-medium transition-colors ${
                  pathname === l.href
                    ? "text-brand-700"
                    : "text-slate-500 hover:text-slate-900"
                }`}
              >
                {l.label}
              </Link>
            )
          )}
          <Link
            href="/dashboard"
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 transition-colors"
          >
            Dashboard
          </Link>
        </div>

        {/* ── Mobile hamburger ── */}
        <button
          onClick={() => setOpen(!open)}
          className="md:hidden p-2 rounded-lg text-slate-600 hover:bg-brand-50"
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* ── Mobile drawer ── */}
      {open && (
        <div className="md:hidden border-t border-slate-100 bg-white shadow-lg">
          <div className="space-y-1 px-5 py-4">
            {links.map((l) =>
              l.scroll ? (
                <button
                  key={l.label}
                  onClick={() =>
                    pathname === "/"
                      ? handleScroll(l.href.replace("/", ""))
                      : (window.location.href = l.href)
                  }
                  className="block w-full text-left rounded-lg px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-brand-50 hover:text-brand-700 transition-colors"
                >
                  {l.label}
                </button>
              ) : (
                <Link
                  key={l.label}
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className={`block rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${
                    pathname === l.href
                      ? "bg-brand-50 text-brand-700"
                      : "text-slate-600 hover:bg-brand-50 hover:text-brand-700"
                  }`}
                >
                  {l.label}
                </Link>
              )
            )}
            <Link
              href="/dashboard"
              onClick={() => setOpen(false)}
              className="mt-2 block rounded-lg bg-brand-600 px-4 py-2.5 text-center text-sm font-semibold text-white hover:bg-brand-700 transition-colors"
            >
              Dashboard
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
