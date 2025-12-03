/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'helios-dark': '#0f172a',
                'helios-card': '#1e293b',
                'helios-accent': '#38bdf8',
                'helios-success': '#22c55e',
                'helios-danger': '#ef4444',
                'helios-warning': '#eab308',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
