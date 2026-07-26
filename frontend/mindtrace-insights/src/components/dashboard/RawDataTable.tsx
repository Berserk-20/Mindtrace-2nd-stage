import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

interface RawDataTableProps {
    data: any[];
}

export function RawDataTable({ data }: RawDataTableProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="relative p-5 rounded-xl card-surface">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between text-xs font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
            >
                <span>📄 Raw Data</span>
                {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {isExpanded && (
                <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-border">
                                <th className="py-2 px-3 text-left text-xs font-mono text-muted-foreground">Timestamp</th>
                                <th className="py-2 px-3 text-left text-xs font-mono text-muted-foreground">Emotion</th>
                                <th className="py-2 px-3 text-left text-xs font-mono text-muted-foreground">Focus Level</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((entry, index) => (
                                <tr key={index} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
                                    <td className="py-2 px-3 text-xs text-muted-foreground font-mono">
                                        {entry.timestamp}
                                    </td>
                                    <td className="py-2 px-3 text-xs text-foreground font-medium">{entry.emotion}</td>
                                    <td className="py-2 px-3 text-xs text-muted-foreground">{(entry.focus_level * 100).toFixed(1)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
