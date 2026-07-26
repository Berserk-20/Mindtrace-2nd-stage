import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen relative overflow-hidden bg-background text-foreground">
      {/* Animated Background */}
      {/* Animated Background - More Colorful */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-50/80 via-background to-cyan-50/80"></div>

      {/* Neural Network Pattern Overlay - Intensified */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-[100px] animate-pulse mix-blend-multiply"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/30 rounded-full blur-[100px] animate-pulse mix-blend-multiply" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-blue-500/20 rounded-full blur-[80px] mix-blend-multiply"></div>
      </div>

      {/* Floating particles effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/40 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${5 + Math.random() * 10}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 5}s`
            }}
          ></div>
        ))}
      </div>

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.05)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0; }
          10% { opacity: 0.4; }
          50% { transform: translateY(-100px) translateX(50px); opacity: 0.6; }
          90% { opacity: 0.4; }
        }
      `}</style>

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="flex items-center justify-between px-6 md:px-10 h-16 bg-background/80 backdrop-blur-xl border-b border-border">
          <div className="flex items-center gap-3">
            <img src="/mindtrace-logo.svg" alt="MindTrace" className="h-8 w-8" />
            <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              MindTrace
            </span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm text-muted-foreground">
            <a href="#about" className="hover:text-foreground transition-colors hover:text-cyan-600">About Us</a>
            <a href="#resources" className="hover:text-foreground transition-colors hover:text-cyan-600">Resources</a>
            <a href="#tutorial" className="hover:text-foreground transition-colors hover:text-cyan-600">Tutorial</a>
          </nav>
          <div className="flex items-center gap-3">
            <Link to="/login">
              <Button variant="outline" className="bg-card border-input text-foreground hover:bg-muted hover:border-purple-500/50 transition-all">
                Log in
              </Button>
            </Link>
            <Button
              className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white hover:from-purple-600 hover:to-cyan-600 shadow-lg shadow-purple-500/30 transition-all"
              onClick={() => navigate("/signup")}
            >
              Get started
            </Button>
          </div>
        </header>

        {/* Hero */}
        <section className="px-6 md:px-10 py-20 md:py-32 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/20 backdrop-blur-sm text-sm text-foreground shadow-sm">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-500 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            Introducing MindTrace
          </div>
          <h1 className="mt-8 text-5xl md:text-7xl font-bold tracking-tight text-foreground">
            <span className="bg-gradient-to-r from-foreground via-purple-600 to-cyan-600 bg-clip-text text-transparent">
              Build smarter sessions
            </span>
          </h1>
          <p className="mt-6 text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Real-time behavioral intelligence: emotions, engagement, attention — unified and actionable.
          </p>
          <div className="mt-10 flex items-center justify-center gap-4">
            <Button
              size="lg"
              className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white hover:from-purple-600 hover:to-cyan-600 shadow-xl shadow-purple-500/30 hover:shadow-purple-500/50 transition-all text-base px-8"
              onClick={() => navigate("/signup")}
            >
              Get started
            </Button>
            <a href="#tutorial">
              <Button
                size="lg"
                variant="outline"
                className="bg-card border-input text-foreground hover:bg-muted hover:border-cyan-500/50 backdrop-blur-sm transition-all text-base px-8"
              >
                See tutorial
              </Button>
            </a>
          </div>
        </section>

        {/* About */}
        <section id="about" className="px-6 md:px-10 py-16 border-t border-purple-100 bg-purple-50/30 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-cyan-600 bg-clip-text text-transparent">
              About Us
            </h2>
            <p className="mt-4 text-muted-foreground text-lg leading-relaxed">
              MindTrace helps you understand user state during live sessions. We analyze facial signals to derive focus,
              emotions, and engagement — streaming metrics to dashboards in real time.
            </p>
          </div>
        </section>

        {/* Resources */}
        <section id="resources" className="px-6 md:px-10 py-16 border-t border-border">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-cyan-600 bg-clip-text text-transparent">
              Resources
            </h2>
            <div className="mt-8 grid md:grid-cols-3 gap-6">
              <a href="https://mindtrace-insights.vercel.app" target="_blank" rel="noopener noreferrer" className="group p-6 rounded-xl border border-purple-500/20 bg-gradient-to-br from-purple-500/5 to-transparent hover:from-purple-500/10 hover:border-purple-400/40 transition-all backdrop-blur-sm">
                <p className="font-semibold text-lg text-purple-600">🚀 Live App</p>
                <p className="text-sm text-muted-foreground mt-2 leading-relaxed">Access MindTrace directly — no installation needed. Sign up and start a session instantly.</p>
              </a>
              <a href="https://huggingface.co/spaces/GaikwadSankalp-1424/mindtrace-backend" target="_blank" rel="noopener noreferrer" className="group p-6 rounded-xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/5 to-transparent hover:from-cyan-500/10 hover:border-cyan-400/40 transition-all backdrop-blur-sm">
                <p className="font-semibold text-lg text-cyan-600">⚙️ Backend API</p>
                <p className="text-sm text-muted-foreground mt-2 leading-relaxed">FastAPI backend powered by ResNet18 + MediaPipe, deployed on Hugging Face Spaces.</p>
              </a>
              <a href="https://github.com/Berserk-20" target="_blank" rel="noopener noreferrer" className="group p-6 rounded-xl border border-blue-500/20 bg-gradient-to-br from-blue-500/5 to-transparent hover:from-blue-500/10 hover:border-blue-400/40 transition-all backdrop-blur-sm">
                <p className="font-semibold text-lg text-blue-600">💻 GitHub</p>
                <p className="text-sm text-muted-foreground mt-2 leading-relaxed">Explore the full source code — frontend, backend, and trained emotion detection model.</p>
              </a>
            </div>
          </div>
        </section>

        {/* Tutorial */}
        <section id="tutorial" className="px-6 md:px-10 py-16 border-t border-cyan-100 bg-cyan-50/30 backdrop-blur-sm mb-12">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-cyan-600 bg-clip-text text-transparent">
              Tutorial
            </h2>
            <ol className="mt-6 space-y-4 text-muted-foreground">
              <li className="flex gap-3">
                <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-600 font-semibold text-sm">1</span>
                <span className="pt-1">Visit <a href="https://mindtrace-insights.vercel.app" className="text-cyan-600 underline underline-offset-2 hover:text-cyan-500">mindtrace-insights.vercel.app</a> and <span className="font-medium text-foreground">Sign Up</span> or Log in.</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-600 font-semibold text-sm">2</span>
                <span className="pt-1"><span className="font-medium text-foreground">Allow camera access</span> when the browser prompts — MindTrace uses your webcam for real-time emotion detection.</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-600 font-semibold text-sm">3</span>
                <span className="pt-1">Navigate to <span className="font-medium text-foreground">Live Session</span> and click <span className="font-medium text-foreground">Start</span> — your face will be detected and emotions labeled in real time.</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-600 font-semibold text-sm">4</span>
                <span className="pt-1">Use <span className="font-medium text-foreground">Pause / Resume / Stop</span> to control the session. Your data is saved automatically.</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-600 font-semibold text-sm">5</span>
                <span className="pt-1">After stopping, visit the <span className="font-medium text-foreground">Dashboard</span> to review your emotion timeline, engagement scores, and full session history.</span>
              </li>
            </ol>
          </div>
        </section>
      </div>
    </div>
  );
}
