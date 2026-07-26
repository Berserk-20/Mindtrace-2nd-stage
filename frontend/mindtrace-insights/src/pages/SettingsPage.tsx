import { useEffect, useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { Switch } from "@/components/ui/switch";
import { fetchUserProfile } from "@/lib/api";
import {
  User,
  Bell,
  Shield,
  Zap,
  Globe,
  Mail,
  Cpu,
  Lock
} from "lucide-react";

const SettingsPage = () => {
  const [userProfile, setUserProfile] = useState<{ name: string, email: string } | null>(null);

  useEffect(() => {
    const loadProfile = async () => {
      const profile = await fetchUserProfile();
      if (profile) {
        setUserProfile(profile);
      }
    };
    loadProfile();
  }, []);

  return (
    <div className="min-h-screen pb-10">
      <TopBar title="Settings" subtitle="Configure MindTrace preferences & system" />

      <div className="p-6 max-w-6xl mx-auto space-y-6">

        {/* Profile Card - Full Width */}
        <div className="relative overflow-hidden rounded-xl border border-border bg-gradient-to-br from-purple-500/10 via-background to-background p-6 shadow-sm">
          <div className="absolute top-0 right-0 -mt-4 -mr-4 h-32 w-32 rounded-full bg-purple-500/20 blur-3xl"></div>

          <div className="flex items-center gap-6 relative z-10">
            <div className="h-20 w-20 rounded-full bg-gradient-to-br from-purple-500 to-cyan-500 p-[2px]">
              <div className="h-full w-full rounded-full bg-background flex items-center justify-center overflow-hidden">
                <User className="h-10 w-10 text-muted-foreground" />
              </div>
            </div>

            <div className="space-y-1">
              {userProfile ? (
                <>
                  <div className="flex items-center gap-2 text-sm text-foreground font-medium">
                    <Mail className="h-4 w-4 text-primary" />
                    <span>{userProfile.email}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-[10px] font-mono uppercase tracking-wider border border-primary/20">
                      Pro Plan
                    </span>
                  </div>
                </>
              ) : (
                <div className="space-y-2">
                  <div className="h-6 w-48 bg-muted rounded animate-pulse" />
                  <div className="h-4 w-32 bg-muted/50 rounded animate-pulse" />
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* General Settings */}
          <div className="card-surface p-6 space-y-6 hover:shadow-purple-500/5 hover:border-purple-500/30 transition-all duration-300">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-500">
                <Bell className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-foreground">Notifications</h3>
            </div>

            <div className="space-y-5">
              <div className="flex items-center justify-between group">
                <div className="space-y-0.5">
                  <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">Push Notifications</p>
                  <p className="text-xs text-muted-foreground">Alerts for engagement drops & spikes</p>
                </div>
                <Switch defaultChecked className="data-[state=checked]:bg-blue-500" />
              </div>

              <div className="w-full h-px bg-border" />

              <div className="flex items-center justify-between group">
                <div className="space-y-0.5">
                  <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">Email Digests</p>
                  <p className="text-xs text-muted-foreground">Weekly session analytics summary</p>
                </div>
                <Switch />
              </div>
            </div>
          </div>

          {/* Privacy & Data */}
          <div className="card-surface p-6 space-y-6 hover:shadow-emerald-500/5 hover:border-emerald-500/30 transition-all duration-300">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500">
                <Shield className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-foreground">Privacy & Data</h3>
            </div>

            <div className="space-y-5">
              <div className="flex items-center justify-between group">
                <div className="space-y-0.5">
                  <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">Store Raw Video</p>
                  <p className="text-xs text-muted-foreground">Save webcam recordings with data</p>
                </div>
                <Switch className="data-[state=checked]:bg-emerald-500" />
              </div>

              <div className="w-full h-px bg-border" />

              <div className="flex items-center justify-between group">
                <div className="space-y-0.5">
                  <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">Anonymize Data</p>
                  <p className="text-xs text-muted-foreground">Remove PII from exported reports</p>
                </div>
                <Switch defaultChecked className="data-[state=checked]:bg-emerald-500" />
              </div>
            </div>
          </div>

          {/* Automation */}
          <div className="card-surface p-6 space-y-6 hover:shadow-orange-500/5 hover:border-orange-500/30 transition-all duration-300">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-orange-500/10 text-orange-500">
                <Zap className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-foreground">Automation</h3>
            </div>

            <div className="space-y-5">
              <div className="flex items-center justify-between group">
                <div className="space-y-0.5">
                  <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">Auto-Start Session</p>
                  <p className="text-xs text-muted-foreground">Begin tracking on webcam detection</p>
                </div>
                <Switch className="data-[state=checked]:bg-orange-500" />
              </div>

              <div className="w-full h-px bg-border" />

              <div className="space-y-3">
                <p className="text-sm font-medium text-foreground">Alert Thresholds</p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded bg-muted/30 border border-border">
                    <span className="text-xs text-muted-foreground block mb-1">Distraction</span>
                    <span className="text-sm font-mono text-orange-500">8s</span>
                  </div>
                  <div className="p-3 rounded bg-muted/30 border border-border">
                    <span className="text-xs text-muted-foreground block mb-1">Engagement</span>
                    <span className="text-sm font-mono text-orange-500">&lt; 45%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* API Config */}
          <div className="card-surface p-6 space-y-6 hover:shadow-cyan-500/5 hover:border-cyan-500/30 transition-all duration-300">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-500">
                <Globe className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-foreground">API Configuration</h3>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">REST Endpoint</p>
                  <div className="flex items-center gap-1 text-[10px] text-success">
                    <div className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                    Online
                  </div>
                </div>
                <div className="h-10 px-3 flex items-center rounded-md border border-border bg-muted text-xs font-mono text-cyan-600 shadow-inner">
                  https://api.mindtrace.ai/v1
                </div>
              </div>

              <div>
                <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1.5">WebSocket Feed</p>
                <div className="h-10 px-3 flex items-center rounded-md border border-border bg-muted text-xs font-mono text-purple-600 shadow-inner">
                  wss://ws.mindtrace.ai/live
                </div>
              </div>

              <div className="pt-2">
                <button className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1">
                  <Lock className="h-3 w-3" />
                  Regenerate API Keys
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
