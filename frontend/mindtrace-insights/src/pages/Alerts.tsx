import { useEffect, useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { AlertTriangle, Info, CheckCircle, Zap, Eye, Clock } from "lucide-react";
import { fetchAlerts } from "@/lib/api";

const iconMap = {
  warning: AlertTriangle,
  error: AlertTriangle,
  info: Info,
  success: CheckCircle,
};

const colorMap = {
  warning: "text-warning border-l-warning",
  error: "text-destructive border-l-destructive",
  info: "text-primary border-l-primary",
  success: "text-success border-l-success",
};

const Alerts = () => {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAlerts = async () => {
      try {
        const data = await fetchAlerts();
        if (data) {
          setAlerts(data);
        }
      } catch (err) {
        console.error("Failed to load alerts", err);
      } finally {
        setLoading(false);
      }
    };

    loadAlerts();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen text-foreground p-6">
        <TopBar title="Alerts & Events" subtitle="System notifications and behavioral alerts" />
        <div className="flex items-center justify-center h-64">
          <p className="text-gray-400">Loading alerts...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <TopBar title="Alerts & Events" subtitle="System notifications and behavioral alerts" />

      <div className="p-6 space-y-3">
        {alerts.length === 0 ? (
          <div className="text-center py-10 text-gray-400">No alerts found.</div>
        ) : (
          alerts.map((alert) => {
            const Icon = iconMap[alert.type as keyof typeof iconMap] || Info;
            const colors = colorMap[alert.type as keyof typeof colorMap] || colorMap.info;
            return (
              <div key={alert.id} className={`card-surface border-l-2 ${colors.split(" ")[1]} p-4`}>
                <div className="flex items-start gap-3">
                  <Icon className={`h-4 w-4 mt-0.5 shrink-0 ${colors.split(" ")[0]}`} />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="text-sm font-medium text-foreground">{alert.title}</h4>
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] font-mono text-muted-foreground">{alert.session}</span>
                        <span className="text-[10px] font-mono text-muted-foreground">{alert.time}</span>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{alert.description}</p>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default Alerts;
