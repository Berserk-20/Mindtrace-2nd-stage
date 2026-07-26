import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Maximize2 } from "lucide-react";
import { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

interface EmotionTimelineProps {
  data: Array<{
    timestamp: string;
    emotion: string;
    focus_level: number;
  }>;
}

// Map emotions to numeric values for Y-axis positioning
const EMOTION_MAP: Record<string, number> = {
  "Neutral": 0,
  "Happy": 1,
  "Angry": 2,
  "Sad": 3,
  "Fear": 4,
  "Surprise": 5,
  "Disgust": 6
};

// Color mapping for each emotion
const EMOTION_COLORS: Record<string, string> = {
  "Neutral": "#9ca3af",
  "Happy": "#10b981",
  "Angry": "#ef4444",
  "Sad": "#3b82f6",
  "Fear": "#f59e0b",
  "Surprise": "#ec4899",
  "Disgust": "#8b5cf6"
};

export function EmotionTimeline({ data }: EmotionTimelineProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Transform data for scatter chart
  const chartData = data.map((entry) => ({
    timestamp: entry.timestamp, // Already session-relative (00:00, 00:30, etc.)
    emotion_code: EMOTION_MAP[entry.emotion] || 0,
    emotion: entry.emotion,
    focus_level: entry.focus_level
  }));

  const ChartComponent = ({ height = 300 }: { height?: number }) => {
    // Calculate minimum width based on data points to enable scrolling
    const minWidth = Math.max(1200, data.length * 20); // At least 20px per data point

    return (
      <div className="w-full overflow-x-auto">
        <div style={{ minWidth: `${minWidth}px`, width: '100%' }}>
          <ResponsiveContainer width="100%" height={height}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="timestamp"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
                tickFormatter={(value) => {
                  // Only show labels at 5-second intervals (00:00, 00:05, 00:10, etc.)
                  const [mins, secs] = value.split(':').map(Number);
                  const totalSeconds = mins * 60 + secs;
                  return totalSeconds % 5 === 0 ? value : '';
                }}
                label={{
                  value: "Session Time (MM:SS)",
                  position: "insideBottom",
                  offset: -5,
                  fill: "hsl(var(--muted-foreground))",
                  style: { fontSize: 14, fontWeight: 'bold' }
                }}
              />
              <YAxis
                dataKey="emotion_code"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 13, fontWeight: 500 }}
                ticks={[0, 1, 2, 3, 4, 5, 6]}
                tickFormatter={(value) => {
                  const emotions = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Surprise", "Disgust"];
                  return emotions[value] || "";
                }}
                domain={[-0.5, 6.5]}
                width={75}
              />
              <Tooltip
                content={({ payload }) => {
                  if (!payload || payload.length === 0) return null;
                  const data = payload[0].payload;
                  return (
                    <div className="bg-popover text-popover-foreground border border-border rounded-lg p-3 shadow-xl">
                      <p className="text-xs font-mono text-muted-foreground mb-1">⏱️ {data.timestamp}</p>
                      <p className="text-base font-bold" style={{ color: EMOTION_COLORS[data.emotion] }}>
                        {data.emotion}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Focus: {(data.focus_level * 100).toFixed(1)}%
                      </p>
                    </div>
                  );
                }}
              />
              <Scatter data={chartData} shape="circle">
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={EMOTION_COLORS[entry.emotion] || "#9ca3af"}
                    fillOpacity={0.95}
                    r={10}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="relative p-5 rounded-xl card-surface">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
            📊 Emotion Timeline
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(true)}
            className="h-7 px-2 text-muted-foreground hover:text-foreground hover:bg-muted/50"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
        </div>
        <ChartComponent />
      </div>

      {/* Expanded View Modal */}
      <Dialog open={isExpanded} onOpenChange={setIsExpanded}>
        <DialogContent className="max-w-6xl w-[95vw] h-[90vh] bg-background/95 backdrop-blur-xl border-border">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <DialogTitle className="text-xl font-mono text-foreground">Emotion Timeline - Expanded View</DialogTitle>
              <p className="text-xs text-muted-foreground">💡 Scroll horizontally to see all data →</p>
            </div>
          </DialogHeader>
          <div className="flex-1 overflow-hidden bg-muted/30 rounded-lg p-4">
            <ChartComponent height={600} />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
