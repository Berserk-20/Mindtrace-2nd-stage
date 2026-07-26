import { AlertTriangle, Info, CheckCircle } from "lucide-react";

interface InsightCardProps {
  type: "warning" | "info" | "success";
  title: string;
  description: string;
  timestamp: string;
}

const icons = {
  warning: AlertTriangle,
  info: Info,
  success: CheckCircle,
};

const styles = {
  warning: {
    border: "border-l-amber-500",
    icon: "text-amber-400",
    glow: "shadow-amber-500/20"
  },
  info: {
    border: "border-l-cyan-500",
    icon: "text-cyan-400",
    glow: "shadow-cyan-500/20"
  },
  success: {
    border: "border-l-emerald-500",
    icon: "text-emerald-400",
    glow: "shadow-emerald-500/20"
  },
};

export function InsightCard({ type, title, description, timestamp }: InsightCardProps) {
  const Icon = icons[type];
  const style = styles[type];

  return (
    <div className={`relative p-4 rounded-lg bg-white/5 border-l-4 ${style.border} border-r border-t border-b border-white/10 backdrop-blur-xl hover:bg-white/10 transition-all duration-300 hover:shadow-lg ${style.glow}`}>
      <div className="flex items-start gap-3">
        <Icon className={`h-5 w-5 mt-0.5 shrink-0 ${style.icon}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1.5">
            <h4 className="text-sm font-semibold text-white">{title}</h4>
            <span className="text-[10px] font-mono text-white/40">{timestamp}</span>
          </div>
          <p className="text-xs text-white/70 leading-relaxed">{description}</p>
        </div>
      </div>
    </div>
  );
}
