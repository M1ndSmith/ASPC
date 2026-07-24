/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  transpilePackages: ["react-plotly.js"],
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
