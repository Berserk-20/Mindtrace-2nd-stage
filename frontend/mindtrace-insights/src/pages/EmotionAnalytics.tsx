import { useEffect, useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { emotionDistribution as initialEmotionDistribution, emotionTransitions as initialEmotionTransitions } from "@/lib/mockData";
import { fetchMetrics } from "@/lib/api";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { ArrowRight } from "lucide-react";

const emotionDescriptions: Record<string, string> = {
  Joy: "Associated with positive engagement and intrinsic motivation. Correlates with higher retention rates.",
  Focus: "Deep concentration state. Optimal for learning and complex task completion.",
  Neutral: "Baseline emotional state. May indicate passive engagement or habitual behavior.",
  Surprise: "Heightened alertness. Often precedes curiosity or confusion depending on context.",
  Frustration: "Cognitive friction state. May signal difficulty or poor UX. Monitor for disengagement.",
  Happy: "Positive emotional state. Indicates satisfaction and enjoyment.",
  Sad: "Low arousal negative state. May indicate disengagement or disappointment.",
  Fear: "High arousal negative state. Response to perceived threat or anxiety.",
  Disgust: "Strong aversion response. May indicate offensive content or bad UX.",
  Angry: "Active hostility or annoyance. Strong negative reaction to stimuli.",
};

const EmotionAnalytics = () => {
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    const loadMetrics = async () => {
      const data = await fetchMetrics();
      if (data) {
        setMetrics(data);
      }
    };

    loadMetrics();
    const interval = setInterval(loadMetrics, 1000); // Poll every second
    return () => clearInterval(interval);
  }, []);

  // Use real data from API if available and not empty, otherwise fall back to initial/mock data for demo
  const emotionDistribution = (metrics?.emotionDistribution && metrics.emotionDistribution.length > 0)
    ? metrics.emotionDistribution
    : initialEmotionDistribution;
  const emotionTransitions = metrics?.emotionTransitions || initialEmotionTransitions;

  // Per-session dominant emotion bar data (Real data from API)
  const sessionEmotions = metrics?.sessionEmotions || [];

  return (
    <div className="min-h-screen">
      <TopBar title="Emotion Analytics" subtitle="Emotion patterns and psychological insights" />

      <div className="p-6 space-y-6">
        <div className="grid grid-cols-3 gap-4">
          {/* Distribution Pie */}
          <div className="card-surface p-4">
            <h3 className="data-label mb-4">Emotion Distribution</h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={emotionDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    dataKey="value"
                    stroke="none"
                  >
                    {emotionDistribution.map((entry: any, i: number) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "6px",
                      fontSize: "11px",
                      color: "hsl(var(--popover-foreground))",
                    }}
                    formatter={(value: number, name: string) => [`${value}%`, name]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-1.5 mt-2">
              {emotionDistribution.map((e: any) => (
                <div key={e.emotion} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-2 rounded-full" style={{ backgroundColor: e.color }} />
                    <span className="text-xs text-foreground">{e.emotion}</span>
                  </div>
                  <span className="text-xs font-mono text-muted-foreground">{e.value}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Per-Session Stacked Bar */}
          <div className="col-span-2 card-surface p-4">
            <h3 className="data-label mb-4">Dominant Emotion Per Session</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sessionEmotions} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="session"
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    axisLine={false}
                    tickLine={false}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "6px",
                      fontSize: "11px",
                      color: "hsl(var(--popover-foreground))",
                    }}
                  />
                  <Bar dataKey="Focus" stackId="a" fill="hsl(187, 80%, 48%)" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="Joy" stackId="a" fill="hsl(152, 60%, 45%)" />
                  <Bar dataKey="Neutral" stackId="a" fill="hsl(215, 15%, 65%)" />
                  <Bar dataKey="Frustration" stackId="a" fill="hsl(0, 72%, 51%)" />
                  <Bar dataKey="Surprise" stackId="a" fill="hsl(38, 92%, 50%)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Transitions + Tooltips */}
        <div className="grid grid-cols-2 gap-4">
          {/* Transitions */}
          <div className="card-surface p-4">
            <h3 className="data-label mb-4">Emotion Transitions</h3>
            <div className="space-y-2">
              {emotionTransitions.map((t: any, i: number) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-2 px-3 rounded-md hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-foreground font-medium">{t.from}</span>
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-foreground font-medium">{t.to}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-20 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary/60"
                        style={{ width: `${(t.count / 42) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-muted-foreground w-6 text-right">{t.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Psychological Meanings */}
          <div className="card-surface p-4">
            <h3 className="data-label mb-4">Emotion Distribution</h3>
            <div className="space-y-3">
              {emotionDistribution.map((e: any) => {
                const desc = emotionDescriptions[e.emotion] || "No description available for this emotion.";
                return (
                  <div key={e.emotion} className="p-3 rounded-md bg-muted/30">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: e.color }} />
                        <span className="text-sm font-medium text-foreground">{e.emotion}</span>
                      </div>
                      <span className="text-xs font-mono text-muted-foreground">{e.value}%</span>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionAnalytics;
