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
          bg: "#080B10",
          panel: "#0D1219",
          border: "#1A2332",
          muted: "#6B7A90",
          text: "#E8EEF7",
          cyan: "#00D4FF",
          "cyan-dim": "#0099BB",
          warn: "#F5A623",
          stop: "#FF4D4F",
          ok: "#2ECC71",
        },
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px rgba(0, 212, 255, 0.15)",
      },
    },
  },
  plugins: [],
};

export default config;
