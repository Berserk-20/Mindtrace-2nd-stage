import { useEffect, useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { KPICard } from "@/components/dashboard/KPICard";
import { EmotionTimeline } from "@/components/dashboard/EmotionTimeline";
import { EmotionDistribution } from "@/components/dashboard/EmotionDistribution";
import { InsightsPanel } from "@/components/dashboard/InsightsPanel";
import { RawDataTable } from "@/components/dashboard/RawDataTable";
import { fetchSessions, fetchEmotions, fetchMetrics } from "@/lib/api";

const Dashboard = () => {
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [emotionData, setEmotionData] = useState<any[]>([]);
  const [systemEvents, setSystemEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Load sessions and metrics on mount
  useEffect(() => {
    const loadData = async () => {
      const [sessionsData, metricsData] = await Promise.all([
        fetchSessions(),
        fetchMetrics()
      ]);

      if (sessionsData && sessionsData.length > 0) {
        setSessions(sessionsData);
        // Auto-select most recent session
        setSelectedSession(sessionsData[0].id);
      }

      if (metricsData && metricsData.systemEvents) {
        setSystemEvents(metricsData.systemEvents);
      }

      setLoading(false);
    };

    loadData();
  }, []);

  // Load emotion data when session changes
  useEffect(() => {
    if (!selectedSession) return;

    const loadEmotions = async () => {
      // Don't set full page loading here, maybe just chart loading?
      // For now, keep it simple but careful not to flicker too much
      const data = await fetchEmotions(selectedSession);
      if (data) {
        setEmotionData(data);
      }
    };

    loadEmotions();
  }, [selectedSession]);

  // Calculate KPIs from emotion data
  const focusPercent = emotionData.length > 0
    ? (emotionData.reduce((sum, e) => sum + e.focus_level, 0) / emotionData.length) * 100
    : 0;

  const fatiguePercent = 100 - focusPercent;
  const totalRecords = emotionData.length;

  const kpis = [
    { value: focusPercent.toFixed(1) + "%", label: "🎯 Focus %" },
    { value: fatiguePercent.toFixed(1) + "%", label: "😴 Fatigue %" },
    { value: totalRecords, label: "📊 Total Records" },
  ];

  if (loading && sessions.length === 0) {
    return (
      <div className="min-h-screen">
        <TopBar title="Dashboard" subtitle="Session analytics" />
        <div className="p-6">
          <p className="text-muted-foreground text-center">Loading sessions...</p>
        </div>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="min-h-screen">
        <TopBar title="Dashboard" subtitle="Session analytics" />
        <div className="p-6">
          <div className="relative p-8 rounded-xl card-surface text-center">
            <p className="text-muted-foreground">No sessions available. Complete a tracking session to see analytics.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <TopBar title="Dashboard" subtitle="Session analytics" />

      <div className="p-6 space-y-6">
        {/* Session Selector */}
        <div className="relative">
          <label className="block text-xs font-mono uppercase tracking-wider text-muted-foreground mb-2">
            Select Session
          </label>
          <select
            value={selectedSession}
            onChange={(e) => setSelectedSession(e.target.value)}
            className="w-full md:w-96 px-4 py-3 rounded-lg bg-card border border-border text-foreground font-mono text-sm focus:outline-none focus:border-primary/50 transition-colors"
          >
            {sessions.map((session) => (
              <option key={session.id} value={session.id} className="bg-popover text-popover-foreground">
                {session.id} - {session.date} ({session.duration})
              </option>
            ))}
          </select>
        </div>

        {loading ? (
          <p className="text-muted-foreground text-center">Loading session data...</p>
        ) : (
          <>
            {/* KPI Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {kpis.map((kpi: any, i: number) => (
                <KPICard key={kpi.label} {...kpi} index={i} />
              ))}
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <EmotionDistribution data={emotionData} />
              <EmotionTimeline data={emotionData} />
            </div>

            {/* Insights & System Events Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <InsightsPanel focusPercent={focusPercent} />

              {/* System Events Log */}
              <div className="relative p-5 rounded-xl card-surface">
                <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-4">📜 Recent Activity</h3>
                <div className="space-y-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                  {systemEvents.length > 0 ? (
                    systemEvents.map((event: any) => (
                      <div key={event.id} className="flex flex-col gap-1 p-3 rounded-md bg-muted/30 border border-border/50">
                        <div className="flex justify-between items-center bg-transparent p-0">
                          <span className="text-xs font-semibold text-primary uppercase tracking-wider bg-transparent p-0">{event.type.replace('_', ' ')}</span>
                          <span className="text-[10px] text-muted-foreground font-mono bg-transparent p-0">{new Date(event.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p className="text-sm text-foreground line-clamp-2 bg-transparent p-0">{event.description}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground italic">No recent events logged.</p>
                  )}
                </div>
              </div>
            </div>

            {/* Raw Data */}
            <RawDataTable data={emotionData} />
          </>
        )}
      </div>
    </div >
  );
};

export default Dashboard;
