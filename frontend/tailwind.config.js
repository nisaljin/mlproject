/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'helios-dark': '#050505',
                'helios-card': '#121212',
                'helios-accent': '#FFD700',
                'helios-blue': '#2979FF',
                'helios-success': '#00E676',
                'helios-danger': '#FF1744',
                'helios-warning': '#FFAB00',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
