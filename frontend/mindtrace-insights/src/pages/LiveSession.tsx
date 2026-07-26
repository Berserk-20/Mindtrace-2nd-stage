import { useEffect, useState, useRef } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { liveMetrics as initialLiveMetrics } from "@/lib/mockData";
import { API_BASE_URL, fetchLiveMetrics, startSession, stopSession, pauseSession, resumeSession } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { Radio, Activity, AlertTriangle, Zap, Eye } from "lucide-react";

// Emotion → color mapping (matches RAF-DB classes from background_agent.py)
const EMOTION_COLORS: Record<string, string> = {
  Happy: "#22c55e", // green
  Neutral: "#94a3b8", // slate
  Surprise: "#f59e0b", // amber
  Angry: "#ef4444", // red
  Sad: "#6366f1", // indigo
  Fear: "#a855f7", // purple
  Disgust: "#f97316", // orange
};

const EMOTION_BG: Record<string, string> = {
  Happy: "bg-green-500/20 text-green-400 border-green-500/40",
  Neutral: "bg-slate-500/20 text-slate-400 border-slate-500/40",
  Surprise: "bg-amber-500/20 text-amber-400 border-amber-500/40",
  Angry: "bg-red-500/20 text-red-400 border-red-500/40",
  Sad: "bg-indigo-500/20 text-indigo-400 border-indigo-500/40",
  Fear: "bg-purple-500/20 text-purple-400 border-purple-500/40",
  Disgust: "bg-orange-500/20 text-orange-400 border-orange-500/40",
};

interface AnalysisResult {
  emotion: string;
  focus_score: number;
  face_found: boolean;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number } | null;
}

const LiveSession = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [agentState, setAgentState] = useState<{ running: boolean; paused: boolean }>({ running: false, paused: false });
  const [videoAvailable, setVideoAvailable] = useState(false);
  const [sessionDuration, setSessionDuration] = useState("00:00:00");
  const [lastResult, setLastResult] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

  const videoRef = useRef<HTMLVideoElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null); // hidden, for grabbing frames
  const overlayRef = useRef<HTMLCanvasElement>(null); // visible, drawn on top of video
  const runningRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);

  // Ref for tracking consecutive low focus frames and alert cooldown
  const lowFocusCount = useRef(0);
  const lastAlertTime = useRef(0);

  // Keep runningRef in sync so cleanup closures can read it
  useEffect(() => {
    runningRef.current = agentState.running;
  }, [agentState.running]);

  // Session duration timer (client-side)
  useEffect(() => {
    if (!metrics?.liveMetrics?.sessionStartTime || !agentState.running || agentState.paused) {
      if (!agentState.running) setSessionDuration("00:00:00");
      return;
    }
    const startTime = metrics.liveMetrics.sessionStartTime * 1000; // Unix → ms
    const update = () => {
      const elapsed = Math.max(0, Math.floor((Date.now() - startTime) / 1000));
      const h = Math.floor(elapsed / 3600);
      const m = Math.floor((elapsed % 3600) / 60);
      const s = elapsed % 60;
      setSessionDuration(`${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`);
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [metrics?.liveMetrics?.sessionStartTime, agentState.running, agentState.paused]);

  // Poll /metrics/live every second
  useEffect(() => {
    const load = async () => {
      const data = await fetchLiveMetrics();
      if (data) {
        setMetrics(data);
        if (data.agentRunning !== undefined && data.agentRunning !== runningRef.current) {
          setAgentState(prev => ({ ...prev, running: data.agentRunning }));
        }
      }
    };
    load();
    const id = setInterval(load, 1000);
    return () => {
      clearInterval(id);
      if (runningRef.current) {
        stopSession().catch(() => { });
      }
    };
  }, []);

  // ── Draw overlay function ─────────────────────────────────────────────────
  const drawOverlay = (result: AnalysisResult | null) => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    // Match canvas size to displayed video element
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!result || !result.face_found || !result.bbox) return;

    const { x, y, width, height } = result.bbox;
    const bx = x * overlay.width;
    const by = y * overlay.height;
    const bw = width * overlay.width;
    const bh = height * overlay.height;

    const color = EMOTION_COLORS[result.emotion] ?? "#ffffff";

    // --- Bounding box ---
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.strokeRect(bx, by, bw, bh);
    ctx.shadowBlur = 0;

    // Corner accents (like OpenCV rectangle)
    const cornerLen = Math.min(bw, bh) * 0.12;
    ctx.lineWidth = 3;
    const corners: [number, number, number, number][] = [
      [bx, by, cornerLen, cornerLen],           // top-left
      [bx + bw - cornerLen, by, cornerLen, cornerLen], // top-right (reflected)
      [bx, by + bh - cornerLen, cornerLen, cornerLen], // bottom-left
      [bx + bw - cornerLen, by + bh - cornerLen, cornerLen, cornerLen], // bottom-right
    ];
    corners.forEach(([cx, cy, cl, _]) => {
      const isRight = cx > bx;
      const isBottom = cy > by;
      ctx.beginPath();
      ctx.moveTo(cx, cy + (isBottom ? -cl : cl));
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx + (isRight ? -cl : cl), cy);
      ctx.strokeStyle = color;
      ctx.stroke();
    });

    // --- Label background (above the box) ---
    const emotion = result.emotion;
    const conf = Math.round(result.confidence * 100);
    const label = `${emotion}  ${conf}%`;
    const fontSize = Math.max(13, Math.min(18, bw / 8));
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;

    const textW = ctx.measureText(label).width;
    const padX = 10, padY = 6;
    const labelH = fontSize + padY * 2;
    const labelX = bx;
    const labelY = by - labelH - 2;

    // Background pill
    ctx.fillStyle = color + "cc"; // 80% opacity
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, textW + padX * 2, labelH, 5);
    ctx.fill();

    // Text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, labelX + padX, labelY + labelH - padY - 1);

    // --- Focus score bar (below the box) ---
    const barW = bw;
    const barH = 5;
    const barX = bx;
    const barY = by + bh + 6;
    const fillW = (result.focus_score / 100) * barW;

    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, 3);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(barX, barY, fillW, barH, 3);
    ctx.fill();

    // "Focus" label
    ctx.font = `10px Inter, sans-serif`;
    ctx.fillStyle = "#ffffffaa";
    ctx.fillText(`Focus: ${Math.round(result.focus_score)}`, barX, barY + barH + 12);
  };

  // ── Camera + analysis loop ────────────────────────────────────────────────
  useEffect(() => {
    let active = true;

    if (!agentState.running || agentState.paused) {
      // Stop camera & clear overlay
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      setVideoAvailable(false);
      // Clear overlay
      const overlay = overlayRef.current;
      if (overlay) {
        const ctx = overlay.getContext("2d");
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
      }
      return;
    }

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => { });
        }
        setVideoAvailable(true);

        const loop = async () => {
          if (!active) return;

          const video = videoRef.current;
          const capture = captureRef.current;
          if (video && capture && video.videoWidth > 0) {
            capture.width = video.videoWidth;
            capture.height = video.videoHeight;
            const ctx = capture.getContext("2d");
            if (ctx) {
              ctx.drawImage(video, 0, 0);
              const base64 = capture.toDataURL("image/jpeg", 0.6);

              try {
                const token = localStorage.getItem("token");
                const res = await fetch(`${API_BASE_URL}/analyze_frame`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`,
                    "ngrok-skip-browser-warning": "true"
                  },
                  body: JSON.stringify({ image_base64: base64 })
                });
                if (res.ok) {
                  const result: AnalysisResult = await res.json();
                  setLastResult(result);
                  drawOverlay(result);

                  // LIVE ALERT LOGIC: Trigger notification if focus drops below 40 for 5 seconds (10 frames at 2 FPS)
                  if (result.focus_score < 40 && result.face_found) {
                    lowFocusCount.current += 1;
                    if (lowFocusCount.current >= 10 && (Date.now() - lastAlertTime.current > 15000)) {
                      const alertTitle = "Low Focus Detected";
                      const alertBody = "It feels like you might be tired. Please take a short break to relax and recharge.";

                      toast({
                        title: alertTitle,
                        description: alertBody,
                        variant: "destructive"
                      });

                      // Trigger OS-level desktop notification
                      if ("Notification" in window && Notification.permission === "granted") {
                        new Notification(alertTitle, { body: alertBody });
                      }

                      lowFocusCount.current = 0;
                      lastAlertTime.current = Date.now();
                    }
                  } else if (result.focus_score >= 40) {
                    lowFocusCount.current = 0;
                  }
                }
              } catch (e) {
                console.error("Frame analysis error:", e);
              }
            }
          }

          if (active) setTimeout(loop, 500); // 2 FPS
        };

        loop();
      } catch (err: any) {
        console.error("Camera error:", err);
        setVideoAvailable(false);

        let errorMessage = "Please allow camera access in browser settings.";
        if (err.name === "NotReadableError") {
          errorMessage = "Camera is already in use by another application (like Zoom or another browser).";
        } else if (err.name === "NotFoundError") {
          errorMessage = "No camera device found on this system.";
        } else if (err.message) {
          errorMessage = `Error: ${err.message}. Please check browser settings.`;
        }

        toast({
          title: "Camera access failed",
          description: errorMessage,
          variant: "destructive"
        });
      }
    };

    start();

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
    };
  }, [agentState.running, agentState.paused]);

  // Redraw overlay when window resizes (keeps box aligned with resized video)
  useEffect(() => {
    const onResize = () => drawOverlay(lastResult);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [lastResult]);

  const liveData = metrics?.liveMetrics || initialLiveMetrics;
  const currentEmotion = liveData?.currentEmotion ?? "Neutral";
  const emotionBadge = EMOTION_BG[currentEmotion] ?? EMOTION_BG["Neutral"];

  return (
    <div className="min-h-screen">
      <TopBar title="Live Session" subtitle="Real-time monitoring" />

      <div className="p-6 space-y-6">
        {/* ── Status bar ── */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-success/10 border border-success/20">
            <span className={`h-2 w-2 rounded-full ${agentState.running && !agentState.paused ? "bg-success animate-pulse" : "bg-muted-foreground"}`} />
            <span className="text-xs font-medium text-success">
              {agentState.running ? (agentState.paused ? "Paused" : "Live") : "Stopped"}
            </span>
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            Session Duration: {sessionDuration}
          </span>

          {/* Current emotion badge */}
          {agentState.running && lastResult && (
            <div className={`px-3 py-1 rounded-full border text-xs font-semibold ${emotionBadge}`}>
              {currentEmotion} · {Math.round((lastResult.confidence ?? 0) * 100)}%
            </div>
          )}

          <div className="ml-auto flex items-center gap-2">
            <Button
              size="sm"
              onClick={async () => {
                // Request OS Notification permission if not already granted/denied
                if ("Notification" in window && Notification.permission === "default") {
                  await Notification.requestPermission();
                }

                try {
                  const res = await startSession();
                  if (res && res.agent_running) {
                    setAgentState({ running: true, paused: false });
                    const data = await fetchLiveMetrics();
                    if (data) setMetrics(data);
                    toast({ title: "Session started 🎬" });
                  } else {
                    toast({ title: "Failed to start", description: "Backend response invalid", variant: "destructive" });
                  }
                } catch (e: any) {
                  toast({ title: "Error starting session", description: e.message, variant: "destructive" });
                }
              }}
              disabled={agentState.running && !agentState.paused}
            >
              Start
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={async () => {
                const res = await pauseSession();
                setAgentState({ running: true, paused: res.paused });
                toast({ title: "Session paused ⏸" });
              }}
              disabled={!agentState.running || agentState.paused}
            >
              Pause
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={async () => {
                const res = await resumeSession();
                setAgentState({ running: true, paused: res.paused });
                toast({ title: "Session resumed ▶" });
              }}
              disabled={!agentState.running || !agentState.paused}
            >
              Resume
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={async () => {
                const res = await stopSession();
                setAgentState({ running: false, paused: false });
                setLastResult(null);
                toast({ title: "Session stopped & saved ✓" });
              }}
              disabled={!agentState.running}
            >
              Stop
            </Button>
          </div>
        </div>

        {/* ── Video feed with overlay ── */}
        <div className="card-surface overflow-hidden max-w-5xl mx-auto w-full">
          <div className="aspect-video bg-muted/30 flex items-center justify-center relative">

            {agentState.running ? (
              <>
                {/* Raw webcam feed */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover ${!videoAvailable ? "opacity-0" : ""}`}
                />

                {/* Face detection overlay canvas */}
                <canvas
                  ref={overlayRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ zIndex: 10 }}
                />

                {/* Hidden capture canvas */}
                <canvas ref={captureRef} className="hidden" />

                {/* No face fallback */}
                {videoAvailable && lastResult && !lastResult.face_found && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-background/80 backdrop-blur-sm border border-border rounded-lg px-4 py-2 z-20">
                    <p className="text-xs text-muted-foreground">No face detected – center your face in the frame</p>
                  </div>
                )}

                {!videoAvailable && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <Radio className="h-8 w-8 text-muted-foreground/30 mb-2" />
                    <p className="text-xs text-muted-foreground">Camera unavailable</p>
                    <p className="text-[10px] text-muted-foreground/60 mt-1">Allow camera access in browser</p>
                  </div>
                )}

                {/* REC indicator */}
                <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-background/80 backdrop-blur-sm border border-border z-20">
                  <span className="h-1.5 w-1.5 rounded-full bg-destructive animate-pulse" />
                  <span className="text-[10px] font-mono text-foreground">REC</span>
                </div>

                {/* Live emotion readout (top-right) */}
                {lastResult?.face_found && (
                  <div className="absolute top-3 right-3 z-20 flex flex-col items-end gap-1">
                    <div className={`px-2.5 py-1 rounded border text-xs font-bold ${EMOTION_BG[lastResult.emotion] ?? EMOTION_BG["Neutral"]}`}>
                      {lastResult.emotion}
                    </div>
                    <div className="text-[10px] font-mono text-foreground bg-background/80 px-2 py-0.5 rounded border border-border">
                      Focus {Math.round(lastResult.focus_score)}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center">
                <Radio className="h-12 w-12 text-muted-foreground/30 mx-auto mb-3" />
                <p className="text-sm text-foreground font-medium">Session Not Started</p>
                <p className="text-xs text-muted-foreground mt-1">Click Start to begin monitoring</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Live stats row ── */}
        {agentState.running && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-5xl mx-auto w-full">
            {[
              { label: "Emotion", value: liveData?.currentEmotion ?? "–" },
              { label: "Confidence", value: `${Math.round((liveData?.emotionConfidence ?? 0) * 100)}%` },
              { label: "Engagement", value: `${Math.round(liveData?.engagementLevel ?? 0)}` },
              { label: "Focus", value: lastResult ? `${Math.round(lastResult.focus_score)}` : "–" },
            ].map(({ label, value }) => (
              <div key={label} className="card-surface p-4 text-center">
                <p className="text-xs text-muted-foreground mb-1">{label}</p>
                <p className="text-xl font-bold text-foreground">{value}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveSession;
