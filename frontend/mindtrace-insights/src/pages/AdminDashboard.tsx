
import { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { Users, Radio, Brain, Calendar, BarChart3 } from "lucide-react";

import { API_BASE_URL } from "@/lib/api";

interface AdminStats {
    stats: {
        totalUsers: number;
        totalSessions: number;
        totalEmotions: number;
        activeSessions: number;
    };
    users: Array<{
        user_id: string;
        name: string;
        email: string;
        role: string;
        total_sessions: number;
        total_emotions: number;
        active_sessions: number;
    }>;
}

export function AdminDashboard() {
    const { token } = useAuth();
    const [data, setData] = useState<AdminStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/admin/stats`, {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                });

                if (!response.ok) {
                    throw new Error("Failed to fetch admin stats");
                }

                const jsonData = await response.json();
                setData(jsonData);
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
            } catch (err: any) {
                setError(err.message || "An error occurred");
            } finally {
                setLoading(false);
            }
        };

        if (token) {
            fetchStats();
        }
    }, [token]);

    if (loading) {
        return (
            <div className="p-8 text-foreground">
                <div className="animate-pulse space-y-4">
                    <div className="h-8 w-48 bg-muted rounded"></div>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="h-32 bg-muted/50 rounded-xl"></div>
                        ))}
                    </div>
                    <div className="h-64 bg-muted/50 rounded-xl"></div>
                </div>
            </div>
        );
    }

    if (error) {
        return <div className="p-8 text-red-400">Error: {error}</div>;
    }

    return (
        <div className="p-8 space-y-8 min-h-screen text-foreground">
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-500 to-cyan-500 bg-clip-text text-transparent">
                    Admin Dashboard
                </h1>
                <p className="text-muted-foreground mt-2">System-wide overview and user statistics</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    icon={Users}
                    label="Total Users"
                    value={data?.stats.totalUsers || 0}
                    color="text-purple-400"
                    bgColor="bg-purple-400/10"
                    borderColor="border-purple-400/20"
                />
                <StatCard
                    icon={Radio}
                    label="Total Sessions"
                    value={data?.stats.totalSessions || 0}
                    color="text-cyan-400"
                    bgColor="bg-cyan-400/10"
                    borderColor="border-cyan-400/20"
                />
                <StatCard
                    icon={Brain}
                    label="Emotions Recorded"
                    value={data?.stats.totalEmotions.toLocaleString() || "0"}
                    color="text-pink-400"
                    bgColor="bg-pink-400/10"
                    borderColor="border-pink-400/20"
                />
                <StatCard
                    icon={BarChart3}
                    label="Active Sessions"
                    value={data?.stats.activeSessions || 0}
                    color="text-emerald-400"
                    bgColor="bg-emerald-400/10"
                    borderColor="border-emerald-400/20"
                />
            </div>

            {/* Users Table */}
            <div className="card-surface overflow-hidden">
                <div className="p-6 border-b border-border flex justify-between items-center">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                        <Users className="w-5 h-5 text-purple-500" />
                        Registered Users
                    </h2>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-muted text-muted-foreground text-sm">
                                <th className="px-6 py-4 font-medium">User</th>
                                <th className="px-6 py-4 font-medium">Role</th>
                                <th className="px-6 py-4 font-medium">Sessions</th>
                                <th className="px-6 py-4 font-medium">Emotions</th>
                                <th className="px-6 py-4 font-medium">Status</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {data?.users.map((user) => (
                                <tr key={user.user_id} className="hover:bg-muted/50 transition-colors">
                                    <td className="px-6 py-4">
                                        <div>
                                            <div className="font-medium text-foreground">{user.name}</div>
                                            <div className="text-sm text-muted-foreground">{user.email}</div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span
                                            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${user.role === "admin"
                                                ? "bg-purple-500/10 text-purple-600 border border-purple-500/20"
                                                : "bg-cyan-500/10 text-cyan-600 border border-cyan-500/20"
                                                }`}
                                        >
                                            {user.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-muted-foreground">{user.total_sessions}</td>
                                    <td className="px-6 py-4 text-muted-foreground">{user.total_emotions}</td>
                                    <td className="px-6 py-4">
                                        {user.active_sessions > 0 ? (
                                            <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600 bg-emerald-500/10 px-2 py-1 rounded-full border border-emerald-500/20">
                                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                                Active
                                            </span>
                                        ) : (
                                            <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full border border-border">
                                                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
                                                Offline
                                            </span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function StatCard({ icon: Icon, label, value, color, bgColor, borderColor }: any) {
    return (
        <div className={`p-6 rounded-xl border ${borderColor} ${bgColor} backdrop-blur-sm`}>
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-sm font-medium text-foreground/70">{label}</p>
                    <p className="mt-2 text-3xl font-bold text-foreground">{value}</p>
                </div>
                <div className={`p-3 rounded-lg bg-background/50 ${color}`}>
                    <Icon className="w-6 h-6" />
                </div>
            </div>
        </div>
    );
}
