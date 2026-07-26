import { NavLink } from "@/components/NavLink";
import {
  LayoutDashboard,
  Radio,
  History,
  Brain,
  BarChart3,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
  LogOut,
} from "lucide-react";
import { useState } from "react";
import { useAuth } from "@/context/AuthContext";

const navItems = [
  { title: "Dashboard", to: "/app", icon: LayoutDashboard },
  { title: "Live Session", to: "/app/live", icon: Radio },
  { title: "Session History", to: "/app/history", icon: History },
  { title: "Emotion Analytics", to: "/app/emotions", icon: Brain },
  { title: "Engagement Reports", to: "/app/reports", icon: BarChart3 },
  { title: "Alerts & Events", to: "/app/alerts", icon: Bell },
  { title: "Settings", to: "/app/settings", icon: Settings },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { isAdmin, logout } = useAuth();

  return (
    <aside
      className={`fixed left-0 top-0 h-screen bg-sidebar border-r border-sidebar-border flex flex-col z-40 transition-all duration-300 ${collapsed ? "w-16" : "w-60"
        }`}
    >
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-sidebar-border">
        <img src="/mindtrace-logo.svg" alt="MindTrace" className="h-7 w-7 shrink-0" />
        {!collapsed && (
          <span className="ml-3 text-lg font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            MindTrace
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/app"}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-sidebar-foreground hover:text-sidebar-primary hover:bg-sidebar-accent transition-all duration-200 group"
            activeClassName="bg-sidebar-accent text-sidebar-primary border border-sidebar-border shadow-sm"
          >
            <item.icon className="h-4 w-4 shrink-0 group-hover:text-cyan-400 transition-colors" />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}

        {isAdmin && (
          <NavLink
            to="/admin"
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-sidebar-foreground hover:text-sidebar-primary hover:bg-sidebar-accent transition-all duration-200 group mt-4 border-t border-sidebar-border pt-4"
            activeClassName="bg-sidebar-accent text-sidebar-primary border border-sidebar-border shadow-sm"
          >
            <div className="h-4 w-4 shrink-0 group-hover:text-purple-400 transition-colors flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-shield-alert"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" /><path d="M12 8v4" /><path d="M12 16h.01" /></svg>
            </div>
            {!collapsed && <span>Admin Panel</span>}
          </NavLink>
        )}
      </nav>

      {/* Logout */}
      <div className="p-3 border-t border-sidebar-border mt-auto">
        <button
          onClick={logout}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-red-400/70 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200 group"
        >
          <LogOut className="h-4 w-4 shrink-0 transition-colors" />
          {!collapsed && <span>Logout</span>}
        </button>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="h-12 flex items-center justify-center border-t border-sidebar-border text-sidebar-foreground hover:text-sidebar-primary hover:bg-sidebar-accent transition-all"
      >
        {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
      </button>
    </aside>
  );
}
