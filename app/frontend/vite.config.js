import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend (http://127.0.0.1:8000) allows CORS from http://localhost:5173,
// so the dev server is pinned to that exact origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: 'localhost',
    port: 5173,
    strictPort: true,
  },
});
