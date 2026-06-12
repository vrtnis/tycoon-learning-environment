import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { Banknote, Box, Gauge, Landmark, Pause, Play, RotateCcw, SkipForward, Upload } from "lucide-react";
import { BenchmarkPage } from "./components/BenchmarkPage";
import { GameCanvas } from "./components/GameCanvas";
import { ReplayCharts } from "./components/ReplayCharts";
import { assertReplayManifest, familyLabel, money, observationAt } from "./sim/replay";
import type { Observation, ReplayManifest } from "./sim/types";

const SAMPLE_REPLAY = "/sample-replay.json";

export default function App(): JSX.Element {
  const params = new URLSearchParams(window.location.search);
  if (params.get("view") === "replay" || params.has("replay")) return <ReplayApp />;
  return <BenchmarkPage />;
}

function openReplay(url: string): void {
  const next = new URL(window.location.href);
  next.searchParams.set("view", "replay");
  next.searchParams.set("replay", url);
  window.location.href = `${next.pathname}${next.search}${next.hash}`;
}

function ReplayApp(): JSX.Element {
  const [manifest, setManifest] = useState<ReplayManifest | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [status, setStatus] = useState("Load a replay JSON");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const replay = params.get("replay");
    if (replay) void loadReplayUrl(replay);
  }, []);

  useEffect(() => {
    if (!playing || !manifest) return;
    if (cursor >= manifest.events.length) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setCursor((value) => Math.min(value + 1, manifest.events.length)), 900);
    return () => window.clearTimeout(timer);
  }, [cursor, manifest, playing]);

  const observation = useMemo(() => (manifest ? observationAt(manifest, cursor) : null), [manifest, cursor]);
  const currentEvent = manifest?.events[Math.max(0, Math.min(cursor, manifest.events.length) - 1)] ?? manifest?.events[0];
  const focus = currentEvent?.focus;

  const loadReplayUrl = async (url: string) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Replay fetch failed: ${response.status}`);
    const loaded = assertReplayManifest(await response.json());
    setManifest(loaded);
    setCursor(0);
    setPlaying(false);
    setStatus(`Loaded ${loaded.worldId}`);
  };

  const loadFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const loaded = assertReplayManifest(JSON.parse(await file.text()));
      setManifest(loaded);
      setCursor(0);
      setPlaying(false);
      setStatus(`Loaded ${file.name}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Replay load failed");
    }
  };

  const loadSample = () => {
    loadReplayUrl(SAMPLE_REPLAY).catch((error) => setStatus(error instanceof Error ? error.message : "Sample replay unavailable"));
  };

  const restart = () => {
    setCursor(0);
    setPlaying(false);
  };

  const step = () => {
    if (!manifest) return;
    setCursor((value) => Math.min(value + 1, manifest.events.length));
  };

  const toggle = () => {
    if (!manifest) return;
    if (cursor >= manifest.events.length) {
      setCursor(0);
      setPlaying(true);
      return;
    }
    setPlaying((value) => !value);
  };

  if (!observation) {
    return (
      <main className="empty-shell">
        <section className="loader-panel">
          <h1>TycoonLE Replay</h1>
          <p>{status}</p>
          <div className="loader-actions">
            <label className="button primary">
              <Upload size={16} /> Load Replay
              <input type="file" accept="application/json,.json" onChange={loadFile} />
            </label>
            <button type="button" className="button" onClick={loadSample}>
              <Play size={16} /> Sample
            </button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="side-panel">
        <div className="brand-row">
          <div className="brand-mark">
            <Gauge size={20} />
          </div>
          <div>
            <h1>TycoonLE</h1>
            <p>{observation.world.split} / {familyLabel(observation.world.family)} / seed {observation.world.seed}</p>
          </div>
        </div>

        <section className="score-section">
          <div className="score-line">
            <span>Score</span>
            <strong>{observation.metrics.score.toFixed(1)}</strong>
          </div>
          <div className="score-track">
            <div style={{ width: `${Math.min(100, observation.metrics.score)}%` }} />
          </div>
          <p>{observation.world.objective.label}</p>
        </section>

        <section className="metric-grid">
          <Metric icon={<Banknote size={15} />} label="Cash" value={money(observation.company.cash)} />
          <Metric icon={<Landmark size={15} />} label="Debt" value={money(observation.company.loan)} />
          <Metric icon={<Box size={15} />} label="Cargo" value={Math.round(observation.metrics.cargoDelivered).toLocaleString()} />
          <Metric icon={<Gauge size={15} />} label="Profit" value={money(observation.metrics.operatingProfit)} />
          <Metric icon={<Box size={15} />} label="Routes" value={String(observation.metrics.routeCount)} />
          <Metric icon={<Gauge size={15} />} label="Reliability" value={`${Math.round(observation.metrics.averageReliability * 100)}%`} />
        </section>

        <section className="event-log">
          <h2>Action</h2>
          <p>{currentEvent ? describeAction(currentEvent.action) : "Initial state"}</p>
          <small>{currentEvent?.info.rewardDetails?.diagnostics?.join(", ") || "No diagnostics"}</small>
        </section>

        <section className="event-log">
          <h2>Status</h2>
          <p>{status}</p>
          <small>{cursor}/{manifest?.events.length ?? 0} replayed</small>
        </section>
      </aside>

      <section className="map-column">
        <div className="replay-workbench">
          <GameCanvas observation={observation} focus={focus} />
          {manifest ? <ReplayCharts manifest={manifest} cursor={cursor} onCursorChange={setCursor} /> : null}
        </div>
        <div className="timeline-strip">
          <label className="button primary">
            <Upload size={16} /> Load
            <input type="file" accept="application/json,.json" onChange={loadFile} />
          </label>
          <button type="button" className="button" onClick={loadSample}>
            <Play size={16} /> Sample
          </button>
          <button type="button" className="button" onClick={restart} disabled={!manifest}>
            <RotateCcw size={16} /> Restart
          </button>
          <button type="button" className="button" onClick={toggle} disabled={!manifest}>
            {playing ? <Pause size={16} /> : <Play size={16} />} {cursor >= (manifest?.events.length ?? 0) ? "Replay Again" : "Replay"}
          </button>
          <button type="button" className="button" onClick={step} disabled={!manifest}>
            <SkipForward size={16} /> Step
          </button>
          <input
            className="scrubber"
            type="range"
            min="0"
            max={manifest?.events.length ?? 0}
            value={cursor}
            onChange={(event) => setCursor(Number(event.target.value))}
          />
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: JSX.Element; label: string; value: string }): JSX.Element {
  return (
    <div className="metric">
      <span>{icon}</span>
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function describeAction(action: { type: string; [key: string]: unknown }): string {
  if (action.type === "build_route") return `Build ${action.mode} ${action.cargo} route`;
  if (action.type === "add_vehicle") return `Add vehicle to ${action.routeId}`;
  if (action.type === "wait") return `Wait ${action.months} month(s)`;
  if (action.type === "take_loan") return `Take loan ${money(Number(action.amount ?? 0))}`;
  if (action.type === "repay_loan") return `Repay loan ${money(Number(action.amount ?? 0))}`;
  return "Invalid action";
}
