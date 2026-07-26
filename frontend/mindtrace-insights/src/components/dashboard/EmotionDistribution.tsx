import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

interface EmotionDistributionProps {
    data: any[];
}

const COLORS = {
    "Happy": "#10B981",
    "Sad": "#8B5CF6",
    "Angry": "#EF4444",
    "Fear": "#F59E0B",
    "Surprise": "#EC4899",
    "Neutral": "#6B7280",
    "Disgust": "#84CC16"
};

export function EmotionDistribution({ data }: EmotionDistributionProps) {
    // Count emotions
    const emotionCounts: { [key: string]: number } = {};
    data.forEach((entry) => {
        const emotion = entry.emotion;
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    });

    // Convert to chart data
    const chartData = Object.entries(emotionCounts).map(([emotion, count]) => ({
        name: emotion,
        value: count,
    }));

    return (
        <div className="relative p-5 rounded-xl card-surface transition-all duration-300">
            <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-4">Emotion Distribution</h3>
            <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={chartData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={100}
                            paddingAngle={2}
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[entry.name as keyof typeof COLORS] || "#6B7280"} />
                            ))}
                        </Pie>
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "hsl(var(--popover))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: "8px",
                                fontSize: "12px",
                                color: "hsl(var(--popover-foreground))"
                            }}
                        />
                        <Legend />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
