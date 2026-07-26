

interface KPICardProps {
  label: string;
  value: number | string;
  index?: number;
}

export function KPICard({ label, value, index = 0 }: KPICardProps) {
  return (
    <div
      className={`relative p-5 rounded-xl card-surface animate-fade-up stagger-${index + 1} hover:shadow-md transition-all duration-300 hover:scale-[1.02] group`}
    >
      {/* Gradient border glow on hover */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-purple-500/20 to-cyan-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10 blur-xl"></div>

      <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">{label}</p>
      <div className="flex items-end justify-between">
        <span className="text-3xl font-semibold font-mono tabular-nums text-foreground">{value}</span>
      </div>
    </div>
  );
}
