/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        pc: {
          bg:       '#0a0c10',
          surface:  '#11151c',
          border:   '#222a35',
          match:    '#00ff99',
          mismatch: '#ff4b4b',
          uncertain:'#ffaa00',
          accent:   '#4a9eff',
          muted:    '#8ab4c8',
        },
      },
    },
  },
  plugins: [],
}
