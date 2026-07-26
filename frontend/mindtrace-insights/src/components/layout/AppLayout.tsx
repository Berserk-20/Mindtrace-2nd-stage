import { Sidebar } from "./Sidebar";
import { Outlet } from "react-router-dom";

export function AppLayout() {
  return (
    <div className="flex min-h-screen w-full bg-background">
      <Sidebar />
      <div className="flex-1 ml-60 transition-all duration-300">
        <Outlet />
      </div>
    </div>
  );
}
