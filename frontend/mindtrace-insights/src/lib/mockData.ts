export const kpiData = {
  engagementScore: { value: 78.4, change: +3.2, label: "Engagement Score" },
  focusStability: { value: 92.1, change: -0.8, label: "Focus Stability" },
  emotionVariance: { value: 0.34, change: +0.05, label: "Emotion Variance" },
  distractionIndex: { value: 12.7, change: -2.1, label: "Distraction Index" },
};

export const emotionTimelineData = Array.from({ length: 30 }, (_, i) => ({
  time: `${String(Math.floor(i / 2)).padStart(2, "0")}:${i % 2 === 0 ? "00" : "30"}`,
  joy: 40 + Math.random() * 30,
  focus: 60 + Math.random() * 25,
  surprise: 10 + Math.random() * 20,
  neutral: 30 + Math.random() * 20,
  frustration: 5 + Math.random() * 15,
}));

export const engagementData = Array.from({ length: 24 }, (_, i) => ({
  hour: `${String(i).padStart(2, "0")}:00`,
  engagement: 50 + Math.sin(i / 3) * 30 + Math.random() * 10,
  attention: 60 + Math.cos(i / 4) * 20 + Math.random() * 10,
}));

export const insights = [
  {
    id: "1",
    type: "warning" as const,
    title: "Attention Drop Detected",
    description: "User engagement dropped 34% between 14:20–14:35. Correlates with increased frustration markers.",
    timestamp: "2 min ago",
  },
  {
    id: "2",
    type: "info" as const,
    title: "Peak Focus Window",
    description: "Highest sustained focus observed 09:00–11:30. Recommend scheduling critical tasks in this window.",
    timestamp: "15 min ago",
  },
  {
    id: "3",
    type: "success" as const,
    title: "Engagement Recovery",
    description: "Post-break engagement recovered to 89% within 4 minutes. Break strategy is effective.",
    timestamp: "1 hr ago",
  },
];

export const sessions = [
  { id: "S-1024", date: "2026-02-09", duration: "1h 42m", engagement: 82, dominantEmotion: "Focus", status: "completed" },
  { id: "S-1023", date: "2026-02-08", duration: "2h 15m", engagement: 71, dominantEmotion: "Neutral", status: "completed" },
  { id: "S-1022", date: "2026-02-08", duration: "0h 55m", engagement: 94, dominantEmotion: "Joy", status: "completed" },
  { id: "S-1021", date: "2026-02-07", duration: "1h 30m", engagement: 58, dominantEmotion: "Frustration", status: "flagged" },
  { id: "S-1020", date: "2026-02-07", duration: "3h 02m", engagement: 76, dominantEmotion: "Focus", status: "completed" },
  { id: "S-1019", date: "2026-02-06", duration: "1h 18m", engagement: 89, dominantEmotion: "Joy", status: "completed" },
  { id: "S-1018", date: "2026-02-06", duration: "0h 45m", engagement: 43, dominantEmotion: "Frustration", status: "flagged" },
  { id: "S-1017", date: "2026-02-05", duration: "2h 30m", engagement: 85, dominantEmotion: "Focus", status: "completed" },
];

export const emotionDistribution = [
  { emotion: "Joy", value: 28, color: "hsl(152, 60%, 45%)" },
  { emotion: "Focus", value: 35, color: "hsl(187, 80%, 48%)" },
  { emotion: "Neutral", value: 18, color: "hsl(215, 15%, 70%)" },
  { emotion: "Surprise", value: 8, color: "hsl(38, 92%, 50%)" },
  { emotion: "Frustration", value: 11, color: "hsl(0, 72%, 51%)" },
];

export const emotionTransitions = [
  { from: "Focus", to: "Neutral", count: 42 },
  { from: "Neutral", to: "Focus", count: 38 },
  { from: "Focus", to: "Joy", count: 24 },
  { from: "Joy", to: "Focus", count: 21 },
  { from: "Neutral", to: "Frustration", count: 15 },
  { from: "Frustration", to: "Neutral", count: 12 },
  { from: "Joy", to: "Neutral", count: 18 },
  { from: "Surprise", to: "Joy", count: 9 },
];

export const liveMetrics = {
  currentEmotion: "Focus",
  emotionConfidence: 0.87,
  engagementLevel: 84,
  attentionScore: 91,
  sessionDuration: "00:34:12",
  events: [
    { time: "00:02:15", type: "engagement_start", label: "Session began" },
    { time: "00:08:42", type: "peak_focus", label: "Peak focus detected" },
    { time: "00:15:30", type: "distraction", label: "Brief distraction — 8s" },
    { time: "00:22:10", type: "emotion_shift", label: "Shifted: Neutral → Focus" },
    { time: "00:28:55", type: "peak_focus", label: "Sustained focus zone" },
    { time: "00:31:40", type: "distraction", label: "Micro-distraction — 3s" },
  ],
};
