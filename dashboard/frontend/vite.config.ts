import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8831,
    proxy: {
      '/api': 'http://localhost:8832'
    },
      host: '0.0.0.0',
      allowedHosts: ['ncshpcgpu01', 'ncshpc400','ncshpc400.fda.gov'],
    },
})
