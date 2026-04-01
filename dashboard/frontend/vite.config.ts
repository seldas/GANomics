import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const isDev = command === 'serve'
  const backendTarget = env.VITE_API_URL || (isDev ? 'http://localhost:8832' : 'http://localhost:8832')

  return {
    // Dev: app at http://localhost:8831/
    // Prod/preview: app at /ganomics/
    base: isDev ? '/' : '/ganomics/',

    plugins: [react()],

    server: {
      host: true,
      port: 8831,
      allowedHosts: ['ncshpc400', 'ncshpcgpu01',],
      proxy: {
        // Development:
        // frontend calls /api/...
        // Vite proxies to backend http://localhost:8832/api/...
        '/api': {
          target: backendTarget,
          changeOrigin: true,
          secure: false,
        },

        // Optional: also support /ganomics_api in dev if needed
        '/ganomics_api': {
          target: backendTarget,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/ganomics_api/, '/api'),
        },
      },
    },

    preview: {
      host: true,
      port: 8831,
      allowedHosts: ['ncshpc400', 'ncshpcgpu01', 'ncshpc400.fda.gov'],
      proxy: {
        // Preview/production-like:
        // frontend calls /ganomics_api/...
        // Vite preview proxies to backend /api/...
        '/ganomics_api': {
          target: backendTarget,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/ganomics_api/, '/api'),
        },
      },
    },

    define: {
      // Lets frontend know which prefix to use
      __API_BASE__: JSON.stringify(isDev ? '/api' : '/ganomics_api'),
    },
  }
})