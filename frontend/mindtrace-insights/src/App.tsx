
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import LiveSession from "./pages/LiveSession";
import SessionHistory from "./pages/SessionHistory";
import EmotionAnalytics from "./pages/EmotionAnalytics";
import { AdminRoute } from "./components/auth/AdminRoute";
import { AdminDashboard } from "./pages/AdminDashboard";
import Reports from "./pages/Reports";
import Alerts from "./pages/Alerts";
import SettingsPage from "./pages/SettingsPage";
import NotFound from "./pages/NotFound";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import { AuthProvider, useAuth } from "@/context/AuthContext";
import Landing from "./pages/Landing";

const queryClient = new QueryClient();

const ProtectedRoute = () => {
  const { isAuthenticated } = useAuth();
  // We might want to show a loading spinner here if auth state is being determined
  // but since we check localStorage synchronously in AuthProvider, it should be fine.
  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }
  return <Outlet />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            <Route path="/app" element={<ProtectedRoute />}>
              <Route element={<AppLayout />}>
                <Route index element={<Dashboard />} />
                <Route path="live" element={<LiveSession />} />
                <Route path="history" element={<SessionHistory />} />
                <Route path="emotions" element={<EmotionAnalytics />} />
                <Route path="reports" element={<Reports />} />
                <Route path="alerts" element={<Alerts />} />
                <Route path="settings" element={<SettingsPage />} />
              </Route>
            </Route>

            {/* Admin Route */}
            <Route path="/admin" element={<AdminRoute />}>
              <Route element={<AppLayout />}>
                <Route index element={<AdminDashboard />} />
              </Route>
            </Route>

            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
