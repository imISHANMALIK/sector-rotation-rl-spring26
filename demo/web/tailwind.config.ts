import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg:      "#030b18",
        surface: "#071120",
        card:    "#0b1a2e",
        border:  "#162740",
        xlk:     "#3b82f6",
        xlf:     "#10b981",
        xlv:     "#f59e0b",
        cash:    "#6b7280",
        spy:     "#8b5cf6",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
