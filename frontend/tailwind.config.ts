import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        aspc: {
          bg: "#0B0B0F",
          panel: "#141418",
          elevated: "#1A1A20",
          border: "#2A2A32",
          muted: "#8A8A96",
          text: "#F2F2F4",
          accent: "#E8C547",
          "accent-dim": "#C4A63A",
          "accent-soft": "rgba(232, 197, 71, 0.12)",
          // Keep cyan alias pointing at accent for any leftover class names
          cyan: "#E8C547",
          "cyan-dim": "#C4A63A",
          ok: "#2DD4BF",
          warn: "#F97316",
          stop: "#F04343",
        },
      },
      fontFamily: {
        sans: ["Sora", "IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "1.5rem",
        pill: "9999px",
      },
      boxShadow: {
        glow: "0 0 28px rgba(232, 197, 71, 0.18)",
        card: "0 8px 32px rgba(0, 0, 0, 0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
