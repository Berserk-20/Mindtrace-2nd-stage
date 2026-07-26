import { useEffect, useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import {
  BarChart3,
  TrendingUp,
  Users,
  Clock,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchReports } from "@/lib/api";

const Reports = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadReports = async () => {
      try {
        const reportData = await fetchReports();
        if (reportData) {
          setData(reportData);
        }
      } catch (err) {
        console.error("Failed to load reports", err);
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen text-foreground p-6">
        <TopBar title="Engagement Reports" subtitle="Weekly and trend analysis" />
        <div className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">Loading reports...</p>
        </div>
      </div>
    );
  }

  // Fallback if data is null (e.g. API error)
  if (!data) {
    return (
      <div className="min-h-screen text-foreground p-6">
        <TopBar title="Engagement Reports" subtitle="Weekly and trend analysis" />
        <div className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">No data available.</p>
        </div>
      </div>
    );
  }

  const { weeklyData, summary, hourlyData } = data;

  return (
    <div className="min-h-screen">
      <TopBar title="Engagement Reports" subtitle="Weekly and trend analysis" />

      <div className="p-6 space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { icon: BarChart3, label: "Avg Engagement", value: summary.avgEngagement, sub: "Total Average" },
            { icon: TrendingUp, label: "Trend", value: summary.trend, sub: "vs last week" },
            { icon: Users, label: "Total Sessions", value: summary.totalSessions, sub: "All time" },
            { icon: Clock, label: "Avg Duration", value: summary.avgDuration, sub: "Per session" },
          ].map((item) => (
            <div key={item.label} className="card-surface p-4">
              <div className="flex items-center gap-2 mb-2">
                <item.icon className="h-4 w-4 text-primary" />
                <p className="data-label">{item.label}</p>
              </div>
              <p className="metric-value text-foreground">{item.value}</p>
              <p className="text-[10px] text-muted-foreground mt-1">{item.sub}</p>
            </div>
          ))}
        </div>

        {/* Weekly Trend */}
        <div className="card-surface p-4">
          <h3 className="data-label mb-4">Weekly Engagement Trend</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={weeklyData}>
                <defs>
                  <linearGradient id="weekGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(187, 80%, 48%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(187, 80%, 48%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px", color: "hsl(var(--popover-foreground))" }} />
                <Area type="monotone" dataKey="avg" stroke="hsl(var(--primary))" fill="url(#weekGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Hourly Heatmap */}
        <div className="card-surface p-4">
          <h3 className="data-label mb-4">Avg Engagement by Hour of Day</h3>
          <div className="grid grid-cols-12 gap-1">
            {hourlyData.map((d: any, i: number) => {
              const intensity = d.engagement / 100;
              return (
                <div
                  key={i}
                  className="aspect-square rounded-sm flex items-center justify-center group relative"
                  style={{
                    backgroundColor: `hsl(187, 80%, 48%, ${intensity * 0.7})`,
                  }}
                >
                  <span className="text-[8px] font-mono text-foreground/70">{d.hour.split(":")[0]}</span>
                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded bg-card border border-border text-[10px] font-mono text-foreground opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                    {d.engagement.toFixed(0)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reports;
