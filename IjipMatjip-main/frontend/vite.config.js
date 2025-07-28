import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),
    tailwindcss()
  ],
    server: {
      port: 4001,
      host:'0.0.0.0',
      proxy:{
        '/api': 'http://13.55.21.100:8001'
      }
  },
})
