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
          bg: "#e0e5ec",
          panel: "#f0f2f5",
          elevated: "#d1d9e6",
          border: "#babecc",
          "border-dark": "#a3b1c6",
          muted: "#4a5568",
          text: "#2d3436",
          accent: "#ff4757",
          "accent-dim": "#e03e4c",
          "accent-soft": "rgba(255, 71, 87, 0.12)",
          "accent-fg": "#ffffff",
          cyan: "#ff4757",
          "cyan-dim": "#e03e4c",
          ok: "#22c55e",
          warn: "#f59e0b",
          stop: "#ff4757",
          chassis: "#e0e5ec",
          ink: "#2d3436",
          dark: "#2d3436",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "16px",
        xl: "24px",
        "2xl": "30px",
        card: "16px",
        pill: "9999px",
      },
      boxShadow: {
        card: "8px 8px 16px #babecc, -8px -8px 16px #ffffff",
        floating: "12px 12px 24px #babecc, -12px -12px 24px #ffffff, inset 1px 1px 0 rgba(255,255,255,0.5)",
        pressed: "inset 6px 6px 12px #babecc, inset -6px -6px 12px #ffffff",
        recessed: "inset 4px 4px 8px #babecc, inset -4px -4px 8px #ffffff",
        sharp: "4px 4px 8px rgba(0,0,0,0.15), -1px -1px 1px rgba(255,255,255,0.8)",
        glow: "0 0 10px 2px rgba(255, 71, 87, 0.6)",
        "glow-ok": "0 0 10px 2px rgba(34, 197, 94, 0.6)",
        key: "4px 4px 8px rgba(166,50,60,0.4), -4px -4px 8px rgba(255,100,110,0.4)",
      },
      transitionTimingFunction: {
        mech: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
      },
      minHeight: {
        touch: "48px",
      },
    },
  },
  plugins: [],
};

export default config;
