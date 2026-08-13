/** @type {import('next').NextConfig} */
const apiProxyTarget = (
  process.env.ASPC_API_PROXY_TARGET ||
  process.env.NEXT_PUBLIC_API_PROXY_TARGET ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

const nextConfig = {
  // Docker Compose needs standalone; Vercel provides its own output and breaks with it.
  ...(process.env.VERCEL ? {} : { output: "standalone" }),
  reactStrictMode: true,
  transpilePackages: ["react-plotly.js"],
  // Same-origin /backend/* → ASPC API. Avoids browser CORS to :8000 (Compose + IDE browsers).
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiProxyTarget}/:path*`,
      },
    ];
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "plotly.js/dist/plotly": "plotly.js/dist/plotly.min.js",
    };
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    return config;
  },
};

module.exports = nextConfig;
