import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lennox Capital — Live Dashboard",
  description: "Risk-Aware Deep Q-Network for Dynamic Asset Allocation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
