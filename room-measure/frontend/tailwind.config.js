/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: '#E91E63',
        secondary: '#C2185B',
        background: '#1A1A1A',
        surface: '#2C2C2C',
        'text-primary': '#F5F5F5',
        'text-secondary': '#B0B0B0',
        border: '#444444',
        accent: '#FF4081',
        danger: '#D32F2F',
        'window-fill': '#F8BBD0',
        'window-stroke': '#EC407A',
        'accent-dark': '#AD1457',
        'danger-dark': '#C62828',
      },
    },
  },
  plugins: [],
};
