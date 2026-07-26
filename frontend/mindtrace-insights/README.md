# 🧠 MindTrace Insights (Frontend)

Welcome to the **MindTrace Frontend**! This repository contains the interactive user interface for the MindTrace platform, an advanced real-time emotion and engagement tracking system.

Built with performance and aesthetics in mind, this frontend consumes the MindTrace FastAPI backend to visualize user engagement metrics, live webcam feeds, and historical data beautifully.

## ✨ Features

- **Live Session Dashboard:** View real-time emotion tracking and focus scores dynamically.
- **Interactive Charts:** Beautiful data visualization using Recharts to display engagement trends.
- **Webcam Integration:** Seamlessly capture and stream frames to the backend for analysis.
- **Modern UI:** Built with Tailwind CSS and shadcn/ui for a premium, accessible, and responsive design.
- **Authentication:** Clean and secure login/signup flows.

## 🛠️ Technology Stack

- **Framework:** [React 18](https://react.dev/) + [Vite](https://vitejs.dev/) - For lightning-fast development and optimized builds.
- **Language:** [TypeScript](https://www.typescriptlang.org/) - Ensuring type safety and better developer experience.
- **Styling:** [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework for rapid UI development.
- **Components:** [shadcn/ui](https://ui.shadcn.com/) - Beautifully designed, accessible, and customizable components.
- **Charts:** [Recharts](https://recharts.org/) - Composable charting library built on React components.
- **Data Fetching:** [React Query (@tanstack/react-query)](https://tanstack.com/query/latest) - For powerful asynchronous state management.

## 💻 Local Development Setup

Follow these steps to get the frontend running locally on your machine.

### Prerequisites
- Node.js (v18 or higher recommended)
- npm or yarn

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend/mindtrace-insights
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment:**
   Ensure your backend API URL is correctly pointed to your local backend server (usually `http://localhost:8000`). Create a `.env` file if necessary:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Start the Development Server:**
   ```bash
   npm run dev
   ```
   The application will start with hot-module replacement (HMR). Open `http://localhost:5173` in your browser to view the app.

## 🚀 Building for Production

To create an optimized production build, run:
```bash
npm run build
```
The compiled assets will be output to the `dist` directory, ready to be deployed to platforms like Vercel, Netlify, or AWS S3.

## 🤝 Contributing

When contributing to this frontend, please ensure you use the provided ESLint and Prettier configurations to maintain code quality:
```bash
npm run lint
```
