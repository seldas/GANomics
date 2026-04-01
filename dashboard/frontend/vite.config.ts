import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())

  return {
    base: '/ganomics/',
    plugins: [react()],
    server: {
      host: true, // This allows external access
      allowedHosts: ['ncshpc400','ncshpcgpu01'], // Add this line
      proxy: {
        '/ganomics_api': {
          target: env.VITE_API_URL || 'http://localhost:8831',
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/ganomics_api/, '/api'),
        }
      }
    },
    preview: {
      host: true,
      allowedHosts: ['ncshpc400','ncshpcgpu01','ncshpc400.fda.gov'], // Add this for preview mode as well
    }
  }
})
