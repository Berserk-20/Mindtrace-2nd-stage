import { TopBar } from "@/components/layout/TopBar";
import { sessions as initialSessions } from "@/lib/mockData";
import { fetchMetrics } from "@/lib/api";
import { Clock, ChevronDown, Filter, Play } from "lucide-react";
import { useState, useEffect } from "react";

const SessionHistory = () => {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterEmotion, setFilterEmotion] = useState("all");
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

  const sessions = metrics?.sessions || initialSessions;

  const filtered = filterEmotion === "all"
    ? sessions
    : sessions.filter((s: any) => s.dominantEmotion.toLowerCase() === filterEmotion);

  return (
    <div className="min-h-screen">
      <TopBar title="Session History" subtitle="Browse and review past sessions" />

      <div className="p-6 space-y-4">
        {/* Filters */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Filter className="h-3.5 w-3.5" />
            <span className="text-muted-foreground">Filter:</span>
          </div>
          {["all", "focus", "joy", "neutral", "frustration"].map((f) => (
            <button
              key={f}
              onClick={() => setFilterEmotion(f)}
              className={`px-3 py-1 rounded-full text-xs capitalize transition-colors ${filterEmotion === f
                ? "bg-primary/15 text-primary border border-primary/30"
                : "bg-muted text-muted-foreground hover:text-foreground"
                }`}
            >
              {f}
            </button>
          ))}
        </div>

        {/* Session Table */}
        <div className="card-surface overflow-hidden">
          <div className="grid grid-cols-6 gap-4 px-4 py-2.5 border-b border-border">
            {["Session ID", "Date", "Duration", "Engagement", "Emotion", "Status"].map((h) => (
              <span key={h} className="data-label">{h}</span>
            ))}
          </div>

          {filtered.map((session) => (
            <div key={session.id}>
              <button
                onClick={() => setExpandedId(expandedId === session.id ? null : session.id)}
                className="w-full grid grid-cols-6 gap-4 px-4 py-3 hover:bg-muted/30 transition-colors text-left"
              >
                <span className="text-xs font-mono text-foreground">{session.id}</span>
                <span className="text-xs text-muted-foreground">{session.date}</span>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {session.duration}
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${session.engagement}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-foreground">{session.engagement}%</span>
                </div>
                <span className="text-xs text-foreground">{session.dominantEmotion}</span>
                <div className="flex items-center justify-between">
                  <span
                    className={`text-xs font-mono ${session.status === "flagged" ? "text-warning" : "text-success"
                      }`}
                  >
                    {session.status}
                  </span>
                  <ChevronDown
                    className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${expandedId === session.id ? "rotate-180" : ""
                      }`}
                  />
                </div>
              </button>

              {expandedId === session.id && (
                <div className="px-4 pb-4 border-b border-border animate-fade-up">
                  <div className="rounded-md bg-muted/30 p-4 flex items-center justify-between">
                    <div className="space-y-1">
                      <p className="text-xs text-foreground">
                        Dominant emotion: <span className="font-medium">{session.dominantEmotion}</span>
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Average engagement maintained at {session.engagement}% throughout session.
                      </p>
                    </div>
                    <button className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-primary/10 text-primary text-xs font-medium hover:bg-primary/20 transition-colors">
                      <Play className="h-3 w-3" />
                      Replay Session
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SessionHistory;
