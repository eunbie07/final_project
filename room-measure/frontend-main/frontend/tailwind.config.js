/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        'sans': [
          'Pretendard',
          '-apple-system', 
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Oxygen',
          'Ubuntu',
          'Cantarell',
          'Open Sans',
          'Helvetica Neue',
          'sans-serif'
        ],
      },
      colors: {
        primary: '#ff7e97',        // pink
        secondary: '#e6678a',      // darker pink
        background: '#F9FAFB',     // gray-50
        surface: '#FFFFFF',        // white
        'text-primary': '#111827', // gray-900
        'text-secondary': '#6B7280', // gray-500
        border: '#D1D5DB',         // gray-300
        accent: '#ff7e97',         // pink
        danger: '#EF4444',         // red-500
        'window-fill': '#ffeef2',  // light pink
        'window-stroke': '#ff7e97', // pink
        'accent-dark': '#cc5577',  // dark pink
        'danger-dark': '#DC2626',  // red-600
      },
    },
  },
  plugins: [],
};
