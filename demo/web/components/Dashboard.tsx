"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ChevronDown, ChevronUp, Pause, Play, RefreshCw, TrendingDown, TrendingUp, Zap } from "lucide-react";
import EquityCurve from "./EquityCurve";
import DrawdownChart from "./DrawdownChart";
import SectorPie from "./SectorPie";
import IVZScoreChart from "./IVZScoreChart";
import TrainingHistory from "./TrainingHistory";
import MarketAnalysis from "./MarketAnalysis";

// ── Types ────────────────────────────────────────────────────────────────
export interface SimStep {
  step: number;
  date: string;
  action: number;
  action_name: "XLK" | "XLF" | "XLV" | "CASH";
  override: boolean;
  daily_return: number;
  portfolio: number;
  spy: number;
  total_return: number;
  spy_total: number;
  sortino: number;
  max_drawdown: number;
  q_values: [number, number, number, number];
  state: {
    iv_xlk: number; iv_xlf: number; iv_xlv: number;
    z_xlk: number;  z_xlf: number;  z_xlv: number;
    rv_xlk: number; rv_xlf: number; rv_xlv: number;
  };
  action_counts: { XLK: number; XLF: number; XLV: number; CASH: number };
  done: boolean;
}

interface EvalResults {
  total_return: number;
  sortino: number;
  spy_return: number;
  action_distribution: Record<string, number>;
  n_override: number;
}

interface TrainingData {
  ep_rewards: number[];
  ep_losses: number[];
  ep_epsilons: number[];
}

const SECTOR_COLORS: Record<string, string> = {
  XLK: "#3b82f6", XLF: "#10b981", XLV: "#f59e0b", CASH: "#6b7280",
};

const TABS = ["simulation", "market", "training"] as const;
type Tab = typeof TABS[number];

const SPEEDS = [
  { label: "0.5×", value: 0.15 },
  { label: "1×",   value: 0.08 },
  { label: "2×",   value: 0.04 },
  { label: "5×",   value: 0.01 },
  { label: "Max",  value: 0.0  },
];

// ── MetricCard ────────────────────────────────────────────────────────────
function MetricCard({
  label, value, sub, positive, glow,
}: {
  label: string; value: string; sub?: string; positive?: boolean; glow?: "blue"|"green"|"red"|"amber";
}) {
  const glowClass = glow ? `card-glow-${glow}` : "";
  return (
    <div className={`card ${glowClass} flex flex-col gap-1 min-w-0`}>
      <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold">{label}</p>
      <p className={`text-2xl font-bold tabular-nums number-flip ${positive === true ? "text-emerald-400" : positive === false ? "text-red-400" : "text-white"}`}>
        {value}
      </p>
      {sub && <p className="text-xs text-slate-500 truncate">{sub}</p>}
    </div>
  );
}

// ── ActionBadge ───────────────────────────────────────────────────────────
function ActionBadge({ name, override }: { name: string; override: boolean }) {
  if (override) return (
    <span className="badge-override px-3 py-1 rounded-full text-xs font-bold tracking-wider">
      ⚠ OVERRIDE → CASH
    </span>
  );
  const cls = `badge-${name.toLowerCase()}`;
  return (
    <span className={`${cls} px-3 py-1 rounded-full text-xs font-bold tracking-wider`}>
      {name}
    </span>
  );
}

// ── StatePanel ────────────────────────────────────────────────────────────
function StatePanel({ current }: { current: SimStep | null }) {
  if (!current) {
    return (
      <div className="card flex flex-col items-center justify-center h-full gap-3 text-slate-600">
        <Zap size={32} />
        <p className="text-sm">Run simulation to see live agent state</p>
      </div>
    );
  }

  const { state, q_values, action_name, override, date } = current;

  const features = [
    { label: "IV XLK", val: state.iv_xlk, color: "#3b82f6", max: 0.6 },
    { label: "IV XLF", val: state.iv_xlf, color: "#10b981", max: 0.6 },
    { label: "IV XLV", val: state.iv_xlv, color: "#f59e0b", max: 0.6 },
  ];

  const zscores = [
    { label: "Z XLK", val: state.z_xlk, color: "#3b82f6" },
    { label: "Z XLF", val: state.z_xlf, color: "#10b981" },
    { label: "Z XLV", val: state.z_xlv, color: "#f59e0b" },
  ];

  const actionLabels = ["XLK", "XLF", "XLV", "CASH"];
  const maxQ = Math.max(...q_values);

  return (
    <div className="card flex flex-col gap-4 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-slate-500">Date</p>
          <p className="text-sm font-mono text-slate-200">{date}</p>
        </div>
        <ActionBadge name={action_name} override={override} />
      </div>

      {/* Daily return */}
      <div className="flex items-center gap-2">
        {current.daily_return >= 0
          ? <TrendingUp size={14} className="text-emerald-400" />
          : <TrendingDown size={14} className="text-red-400" />}
        <span className={`text-sm font-bold tabular-nums ${current.daily_return >= 0 ? "text-emerald-400" : "text-red-400"}`}>
          {current.daily_return >= 0 ? "+" : ""}{current.daily_return.toFixed(3)}%
        </span>
        <span className="text-xs text-slate-500">daily return</span>
      </div>

      <hr className="border-border" />

      {/* IV Features */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Implied Volatility</p>
        {features.map(f => (
          <div key={f.label} className="flex items-center gap-2 mb-2">
            <span className="text-[10px] font-mono text-slate-400 w-14">{f.label}</span>
            <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.min(f.val / f.max, 1) * 100}%`, background: f.color }}
              />
            </div>
            <span className="text-[10px] font-mono text-slate-300 w-10 text-right">{f.val.toFixed(3)}</span>
          </div>
        ))}
      </div>

      {/* Z-Scores */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">IV Z-Scores (60d)</p>
        {zscores.map(z => {
          const clipped = Math.max(-3, Math.min(3, z.val));
          const pct = ((clipped + 3) / 6) * 100;
          const danger = Math.abs(z.val) > 2.5;
          return (
            <div key={z.label} className="flex items-center gap-2 mb-2">
              <span className="text-[10px] font-mono text-slate-400 w-14">{z.label}</span>
              <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden relative">
                <div className="absolute left-1/2 w-px h-full bg-slate-600" />
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.abs(clipped) / 3 * 50}%`,
                    marginLeft: clipped >= 0 ? "50%" : `${50 - Math.abs(clipped) / 3 * 50}%`,
                    background: danger ? "#ef4444" : z.color,
                  }}
                />
              </div>
              <span className={`text-[10px] font-mono w-12 text-right ${danger ? "text-red-400 font-bold" : "text-slate-300"}`}>
                {z.val.toFixed(2)}
              </span>
              {danger && <span className="text-[8px] text-red-400">⚠</span>}
            </div>
          );
        })}
        {override && (
          <p className="text-[10px] text-red-400 bg-red-900/20 border border-red-800/40 rounded px-2 py-1 mt-1">
            All z-scores &gt; 2.5 → Safety override triggered
          </p>
        )}
      </div>

      <hr className="border-border" />

      {/* Q-Values */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Q-Values → Decision</p>
        {actionLabels.map((name, i) => {
          const isChosen = name === action_name && !override;
          const isCash = name === "CASH" && override;
          const highlight = isChosen || isCash;
          const pct = maxQ > 0 ? (q_values[i] / maxQ) * 100 : 0;
          return (
            <div key={name} className={`flex items-center gap-2 mb-2 rounded px-1 ${highlight ? "bg-white/5" : ""}`}>
              <span className={`text-[10px] font-mono w-10 ${highlight ? "text-white font-bold" : "text-slate-400"}`}>{name}</span>
              <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${pct}%`, background: highlight ? SECTOR_COLORS[name] : "#334155" }}
                />
              </div>
              <span className={`text-[10px] font-mono w-12 text-right ${highlight ? "text-white font-bold" : "text-slate-400"}`}>
                {q_values[i].toFixed(3)}
              </span>
              {highlight && <span className="text-[10px]">✓</span>}
            </div>
          );
        })}
      </div>

      {/* Realized vols */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Realized Vol (20d)</p>
        {[
          { label: "RV XLK", val: state.rv_xlk, color: "#3b82f6" },
          { label: "RV XLF", val: state.rv_xlf, color: "#10b981" },
          { label: "RV XLV", val: state.rv_xlv, color: "#f59e0b" },
        ].map(rv => (
          <div key={rv.label} className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] font-mono text-slate-400 w-14">{rv.label}</span>
            <div className="flex-1 bg-slate-800 rounded-full h-1 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${Math.min(rv.val / 0.04, 1) * 100}%`, background: rv.color }} />
            </div>
            <span className="text-[10px] font-mono text-slate-300 w-10 text-right">{rv.val.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── TradeLog ─────────────────────────────────────────────────────────────
function TradeLog({ history }: { history: SimStep[] }) {
  const last20 = history.slice(-20).reverse();
  if (last20.length === 0) return null;
  return (
    <div className="card">
      <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">Trade Log</p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-[10px] text-slate-500 border-b border-border">
              <th className="text-left py-1 pr-4">Date</th>
              <th className="text-left py-1 pr-4">Action</th>
              <th className="text-right py-1 pr-4">Daily Ret</th>
              <th className="text-right py-1 pr-4">Portfolio</th>
              <th className="text-right py-1 pr-4">vs SPY</th>
              <th className="text-right py-1 pr-4">Override</th>
            </tr>
          </thead>
          <tbody>
            {last20.map((s) => (
              <tr key={s.step} className="border-b border-border/50 hover:bg-white/2">
                <td className="py-1 pr-4 text-slate-400">{s.date}</td>
                <td className="py-1 pr-4">
                  <span
                    className="px-2 py-0.5 rounded text-[10px] font-bold"
                    style={{
                      background: SECTOR_COLORS[s.action_name] + "33",
                      color: SECTOR_COLORS[s.action_name],
                    }}
                  >
                    {s.action_name}
                  </span>
                </td>
                <td className={`py-1 pr-4 text-right ${s.daily_return >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {s.daily_return >= 0 ? "+" : ""}{s.daily_return.toFixed(3)}%
                </td>
                <td className="py-1 pr-4 text-right text-slate-300">
                  {((s.portfolio - 1) * 100).toFixed(2)}%
                </td>
                <td className={`py-1 pr-4 text-right ${s.total_return > s.spy_total ? "text-emerald-400" : "text-red-400"}`}>
                  {(s.total_return - s.spy_total).toFixed(2)}%
                </td>
                <td className="py-1 pr-4 text-right">
                  {s.override ? <span className="text-amber-400">⚠</span> : <span className="text-slate-700">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────
export default function Dashboard() {
  const [history, setHistory] = useState<SimStep[]>([]);
  const [current, setCurrent] = useState<SimStep | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1);
  const [tab, setTab] = useState<Tab>("simulation");
  const [evalResults, setEvalResults] = useState<EvalResults | null>(null);
  const [trainingData, setTrainingData] = useState<TrainingData | null>(null);
  const [marketData, setMarketData] = useState<Record<string, unknown>[]>([]);
  const esRef = useRef<EventSource | null>(null);

  // Fetch static data once
  useEffect(() => {
    fetch("/api/eval-results").then(r => r.json()).then(setEvalResults).catch(() => {});
    fetch("/api/training-history").then(r => r.json()).then(setTrainingData).catch(() => {});
    fetch("/api/data").then(r => r.json()).then(setMarketData).catch(() => {});
  }, []);

  const start = useCallback(() => {
    esRef.current?.close();
    setHistory([]);
    setCurrent(null);
    setIsRunning(true);
    setIsDone(false);

    const delay = SPEEDS[speedIdx].value;
    const url = `/api/inference/stream?delay=${delay}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e: MessageEvent) => {
      const data: SimStep = JSON.parse(e.data);
      setHistory(prev => [...prev, data]);
      setCurrent(data);
      if (data.done) {
        setIsRunning(false);
        setIsDone(true);
        es.close();
      }
    };
    es.onerror = () => { setIsRunning(false); es.close(); };
  }, [speedIdx]);

  const stop = useCallback(() => {
    esRef.current?.close();
    setIsRunning(false);
  }, []);

  const reset = useCallback(() => {
    esRef.current?.close();
    setHistory([]);
    setCurrent(null);
    setIsRunning(false);
    setIsDone(false);
  }, []);

  useEffect(() => () => { esRef.current?.close(); }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        isRunning ? stop() : start();
      }
      if (e.code === "KeyR" && e.target === document.body) reset();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isRunning, start, stop, reset]);

  // Derived chart data
  const equityData = useMemo(() =>
    history.map(s => ({ date: s.date, agent: s.portfolio, spy: s.spy, override: s.override })),
    [history]
  );
  const drawdownData = useMemo(() =>
    history.map(s => ({ date: s.date, drawdown: s.max_drawdown })),
    [history]
  );
  const zscoreData = useMemo(() =>
    history.map(s => ({
      date: s.date,
      z_xlk: s.state.z_xlk,
      z_xlf: s.state.z_xlf,
      z_xlv: s.state.z_xlv,
    })),
    [history]
  );
  const pieData = useMemo(() => {
    const counts = current?.action_counts ?? { XLK: 0, XLF: 0, XLV: 0, CASH: 0 };
    return [
      { name: "XLK", value: counts.XLK, color: "#3b82f6" },
      { name: "XLF", value: counts.XLF, color: "#10b981" },
      { name: "XLV", value: counts.XLV, color: "#f59e0b" },
      { name: "CASH", value: counts.CASH, color: "#6b7280" },
    ].filter(d => d.value > 0);
  }, [current]);

  // Metrics: use live values if running, else pre-computed
  const sortino  = current?.sortino     ?? (evalResults ? evalResults.sortino : null);
  const totalRet = current?.total_return ?? (evalResults ? evalResults.total_return * 100 : null);
  const maxDD    = current?.max_drawdown ?? null;
  const spyTotal = current?.spy_total    ?? (evalResults ? evalResults.spy_return * 100 : null);
  const alpha    = totalRet != null && spyTotal != null ? totalRet - spyTotal : null;
  const overrides = current?.action_counts
    ? history.filter(s => s.override).length
    : evalResults?.n_override ?? null;

  const progress = history.length;
  const TOTAL = 251;

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* ── Header ── */}
      <header className="gradient-header sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity className="text-blue-400" size={20} />
            <span className="font-bold text-white text-sm tracking-tight">Sector Rotation RL</span>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-[11px] text-slate-500 uppercase tracking-wider">Risk-Aware DQN · 2024 Holdout</span>
        </div>

        <nav className="flex gap-1">
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-1.5 rounded text-xs font-semibold uppercase tracking-wider transition-all ${
                tab === t ? "tab-active" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          {isRunning && (
            <div className="flex items-center gap-1.5 text-emerald-400 text-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              LIVE
            </div>
          )}
          {isDone && <span className="text-xs text-blue-400 font-semibold">COMPLETE</span>}
          <span className="text-xs text-slate-600 font-mono">
            {progress}/{TOTAL} days
          </span>
        </div>
      </header>

      <main className="flex-1 p-4 space-y-4 max-w-[1800px] mx-auto w-full">

        {/* ── Simulation Tab ── */}
        {tab === "simulation" && (
          <>
            {/* Controls */}
            <div className="card flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-2">
                <button
                  onClick={isRunning ? stop : start}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                    isRunning
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30"
                      : "bg-blue-500/20 text-blue-300 border border-blue-500/40 hover:bg-blue-500/30"
                  }`}
                >
                  {isRunning ? <Pause size={14} /> : <Play size={14} />}
                  {isRunning ? "Stop" : isDone ? "Replay" : "Run Simulation"}
                </button>
                <button
                  onClick={reset}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-400 border border-border hover:border-slate-600 hover:text-slate-300 transition-all"
                >
                  <RefreshCw size={12} />
                  Reset
                </button>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider">Speed</span>
                {SPEEDS.map((s, i) => (
                  <button
                    key={s.label}
                    onClick={() => setSpeedIdx(i)}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                      speedIdx === i
                        ? "bg-blue-500/30 text-blue-300 border border-blue-500/50"
                        : "text-slate-500 border border-border hover:border-slate-600 hover:text-slate-300"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              {/* Progress bar */}
              <div className="flex-1 min-w-[200px]">
                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span>Progress</span>
                  <span className="font-mono">{progress} / {TOTAL} days</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-200"
                    style={{ width: `${(progress / TOTAL) * 100}%` }}
                  />
                </div>
              </div>

              <div className="text-[11px] text-slate-600">
                <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px]">Space</kbd> play/stop
                &nbsp;·&nbsp;
                <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px]">R</kbd> reset
              </div>
            </div>

            {/* Metrics row */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <MetricCard
                label="Sortino Ratio"
                value={sortino != null ? sortino.toFixed(2) : "—"}
                sub="Annualised downside-adjusted"
                glow={sortino != null && sortino > 2 ? "blue" : undefined}
              />
              <MetricCard
                label="Total Return"
                value={totalRet != null ? `${totalRet >= 0 ? "+" : ""}${totalRet.toFixed(1)}%` : "—"}
                positive={totalRet != null ? totalRet > 0 : undefined}
                sub="2024 holdout set"
                glow={totalRet != null && totalRet > 0 ? "green" : totalRet != null ? "red" : undefined}
              />
              <MetricCard
                label="Max Drawdown"
                value={maxDD != null ? `${maxDD.toFixed(1)}%` : "—"}
                positive={maxDD != null ? false : undefined}
                sub="Peak-to-trough"
                glow={maxDD != null && maxDD < -5 ? "red" : undefined}
              />
              <MetricCard
                label="SPY Benchmark"
                value={spyTotal != null ? `+${spyTotal.toFixed(1)}%` : "—"}
                sub="Buy-and-hold S&P 500"
              />
              <MetricCard
                label="Alpha"
                value={alpha != null ? `${alpha >= 0 ? "+" : ""}${alpha.toFixed(1)}%` : "—"}
                positive={alpha != null ? alpha > 0 : undefined}
                sub="vs SPY"
                glow={alpha != null && alpha > 0 ? "green" : alpha != null ? "red" : undefined}
              />
              <MetricCard
                label="Overrides"
                value={overrides != null ? String(overrides) : "—"}
                sub="Vasant Dhar safety triggers"
                glow={overrides != null && overrides > 0 ? "amber" : undefined}
              />
            </div>

            {/* Main chart + state panel */}
            <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-4">
              <div className="card min-h-[380px]">
                <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
                  Equity Curve — Agent vs SPY (2024)
                </p>
                <EquityCurve data={equityData} />
              </div>
              <StatePanel current={current} />
            </div>

            {/* Bottom charts */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="card min-h-[260px]">
                <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
                  Running Drawdown
                </p>
                <DrawdownChart data={drawdownData} />
              </div>
              <div className="card min-h-[260px]">
                <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
                  Sector Allocation
                </p>
                <SectorPie data={pieData} currentAction={current?.action_name} />
              </div>
              <div className="card min-h-[260px]">
                <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
                  IV Z-Scores (live)
                </p>
                <IVZScoreChart data={zscoreData} />
              </div>
            </div>

            {/* Trade log */}
            {history.length > 0 && <TradeLog history={history} />}
          </>
        )}

        {/* ── Market Analysis Tab ── */}
        {tab === "market" && (
          <MarketAnalysis data={marketData} />
        )}

        {/* ── Training Tab ── */}
        {tab === "training" && (
          <TrainingHistory data={trainingData} evalResults={evalResults} />
        )}
      </main>
    </div>
  );
}
