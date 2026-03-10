import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
<<<<<<< HEAD
    port: 8831,
  }
=======
    host: '0.0.0.0',
    allowedHosts: ['ncshpcgpu01'],
  },
>>>>>>> 64b06a42822606ca5b82c92259563f6e711ba8d3
})
