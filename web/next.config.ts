import type { NextConfig } from "next";

// The Python side serves the API. Proxying through Next keeps the browser on
// one origin, which avoids CORS entirely and means the console works
// unchanged whether the backend is local or behind a tunnel.
const BACKEND = process.env.VAULT_API ?? "http://127.0.0.1:8899";

const config: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND}/api/:path*` }];
  },
};

export default config;
