/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

