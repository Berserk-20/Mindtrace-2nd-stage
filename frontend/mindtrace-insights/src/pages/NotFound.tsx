import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center relative overflow-hidden bg-[#0a0e1a]">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#1a1f2e] via-[#16202a] to-[#0f1419]"></div>

      {/* Glow effects */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-purple-500/30 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-cyan-500/30 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '1.5s' }}></div>
      </div>

      {/* Grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      {/* Content */}
      <div className="text-center relative z-10 px-6">
        <div className="inline-flex items-center gap-2 mb-8">
          <img src="/mindtrace-logo.svg" alt="MindTrace" className="h-12 w-12" />
          <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            MindTrace
          </span>
        </div>
        <h1 className="mb-4 text-8xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
          404
        </h1>
        <p className="mb-8 text-2xl text-white/80 font-medium">Oops! Page not found</p>
        <p className="mb-8 text-white/60 max-w-md mx-auto">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/">
          <Button className="bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 text-white shadow-lg shadow-purple-500/30 transition-all px-8">
            Return to Home
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
