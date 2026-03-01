import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "DermWise — AI Skin Lesion Analysis",
  description:
    "AI-powered skin lesion classification and clinical report generation using EfficientNet-B0, FAISS RAG, and QLoRA fine-tuned TinyLlama.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="flex min-h-screen flex-col">
        <Navbar />
        {/* pt-16 offsets the fixed navbar height (h-16 = 4rem) */}
        <main className="flex-1 pt-16">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
