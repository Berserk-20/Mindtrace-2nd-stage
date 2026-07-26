
import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";

export const AdminRoute = () => {
    const { isAuthenticated, isAdmin } = useAuth();

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    if (!isAdmin) {
        return <Navigate to="/app" replace />;
    }

    return <Outlet />;
};
