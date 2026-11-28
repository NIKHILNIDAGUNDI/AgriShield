import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { readFileSync } from 'node:fs'
import { defineConfig, type Plugin } from 'vite'

// dashboard_v2/train_results.json is the single source; ship a copy for the no-API fallback.
function trainingResults(): Plugin {
  return {
    name: 'agrishield-training-results',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'train_results.json',
        source: readFileSync(new URL('../train_results.json', import.meta.url)),
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), trainingResults()],
  server: {
    port: 5173,
    proxy: {
      '/v1': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
})
