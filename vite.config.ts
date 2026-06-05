import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // tfjs + mediapipe are large; raise warning limit, no inlining of wasm
  build: {
    chunkSizeWarningLimit: 2000,
  },
});
