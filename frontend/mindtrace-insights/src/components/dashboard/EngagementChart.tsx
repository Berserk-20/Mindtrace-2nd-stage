import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface EngagementChartProps {
  data: any[];
}

export function EngagementChart({ data }: EngagementChartProps) {
  return (
    <div className="relative p-5 rounded-xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 transition-all duration-300">
      <h3 className="text-xs font-mono uppercase tracking-wider text-white/50 mb-4">Engagement vs Attention</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="hour"
              tick={{ fontSize: 11, fill: "rgba(255,255,255,0.5)" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "rgba(255,255,255,0.5)" }}
              axisLine={false}
              tickLine={false}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(0,0,0,0.8)",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: "8px",
                fontSize: "12px",
                backdropFilter: "blur(10px)",
                color: "#fff"
              }}
            />
            <Line type="monotone" dataKey="engagement" stroke="#A855F7" strokeWidth={2.5} dot={{ fill: "#A855F7", r: 4 }} activeDot={{ r: 6 }} />
            <Line type="monotone" dataKey="attention" stroke="#06B6D4" strokeWidth={2.5} dot={{ fill: "#06B6D4", r: 4 }} strokeDasharray="5 5" activeDot={{ r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-6 mt-4">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-purple-500 shadow-lg shadow-purple-500/50" />
          <span className="text-xs text-white/70">Engagement</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50" />
          <span className="text-xs text-white/70">Attention</span>
        </div>
      </div>
    </div>
  );
}
