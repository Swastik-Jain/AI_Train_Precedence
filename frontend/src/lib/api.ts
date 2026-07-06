/**
 * Central API configuration — single source of truth for all backend URLs.
 *
 * Local dev  : Vite proxies /api → http://127.0.0.1:8000 and /ws → ws://127.0.0.1:8000
 *              so API_BASE defaults to '' (relative paths) and WS_BASE auto-derives
 *              from the current page origin.
 *
 * Production : set VITE_API_BASE_URL and VITE_WS_BASE_URL in your .env file.
 *              e.g.  VITE_API_BASE_URL=https://api.myapp.com
 *                    VITE_WS_BASE_URL=wss://api.myapp.com
 */

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
export const WS_BASE   = import.meta.env.VITE_WS_BASE_URL   ?? (API_BASE ? API_BASE.replace(/^http/, 'ws') : '');

export const apiUrl = (path: string) => `${API_BASE}${path}`;
export const wsUrl  = (path: string) => `${WS_BASE}${path}`;
