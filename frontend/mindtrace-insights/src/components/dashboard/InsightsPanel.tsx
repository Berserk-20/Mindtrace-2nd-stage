import { AlertCircle, CheckCircle, AlertTriangle } from "lucide-react";

interface InsightsPanelProps {
    focusPercent: number;
}

export function InsightsPanel({ focusPercent }: InsightsPanelProps) {
    const getInsightConfig = () => {
        if (focusPercent >= 70) {
            return {
                type: "success",
                icon: CheckCircle,
                color: "text-emerald-400",
                bgColor: "bg-emerald-500/10",
                borderColor: "border-emerald-500/30",
                message: "✅ Good focus maintained"
            };
        } else if (focusPercent >= 40) {
            return {
                type: "warning",
                icon: AlertTriangle,
                color: "text-yellow-400",
                bgColor: "bg-yellow-500/10",
                borderColor: "border-yellow-500/30",
                message: "⚠️ Moderate focus detected"
            };
        } else {
            return {
                type: "error",
                icon: AlertCircle,
                color: "text-red-400",
                bgColor: "bg-red-500/10",
                borderColor: "border-red-500/30",
                message: "🚨 Low focus detected"
            };
        }
    };

    const config = getInsightConfig();
    const Icon = config.icon;

    return (
        <div className="relative p-5 rounded-xl card-surface">
            <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-4">⚠️ Insights</h3>

            <div className={`flex items-center gap-3 p-4 rounded-lg ${config.bgColor} border ${config.borderColor}`}>
                <Icon className={`h-6 w-6 ${config.color}`} />
                <p className={`text-sm font-medium ${config.color}`}>{config.message}</p>
            </div>
        </div>
    );
}
