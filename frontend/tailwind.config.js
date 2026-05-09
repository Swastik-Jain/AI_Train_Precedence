/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'electric-violet': '#8B5CF6',
        'orbital-white': '#FFFFFF',
      },
      fontFamily: {
        'instrument': ['"Instrument Sans"', 'sans-serif'],
        'geist': ['"Geist Mono"', 'monospace'],
      },
      borderRadius: {
        '2xl': '24px',
      }
    },
  },
  plugins: [],
}
