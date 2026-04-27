"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMemo } from "react";

interface TrainingData {
  ep_rewards:  number[];
  ep_losses:   number[];
  ep_epsilons: number[];
}

interface EvalResults {
  total_return: number;
  sortino:      number;
  spy_return:   number;
  action_distribution: Record<string, number>;
  n_override:   number;
}

function smooth(arr: number[], w = 20): number[] {
  return arr.map((_, i) => {
    const start = Math.max(0, i - w);
    const slice = arr.slice(start, i + 1);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  });
}

function subsample<T>(arr: T[], n = 400): T[] {
  if (arr.length <= n) return arr;
  const step = Math.ceil(arr.length / n);
  return arr.filter((_, i) => i % step === 0);
}

export default function TrainingHistory({
  data,
  evalResults,
}: {
  data: TrainingData | null;
  evalResults: EvalResults | null;
}) {
  const rewardData = useMemo(() => {
    if (!data) return [];
    const smoothed = smooth(data.ep_rewards, 30);
    return subsample(smoothed.map((v, i) => ({ episode: i + 1, value: v })));
  }, [data]);

  const lossData = useMemo(() => {
    if (!data) return [];
    const smoothed = smooth(data.ep_losses, 30);
    return subsample(smoothed.map((v, i) => ({ episode: i + 1, value: v })));
  }, [data]);

  const epsilonData = useMemo(() => {
    if (!data) return [];
    return subsample(data.ep_epsilons.map((v, i) => ({ episode: i + 1, value: v })));
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-600 text-sm">
        Loading training history...
      </div>
    );
  }

  const chartProps = {
    margin: { top: 4, right: 16, bottom: 4, left: 8 } as const,
  };

  const axisProps = {
    tickLine: false,
    axisLine: false,
    tick: { fill: "#475569", fontSize: 10 },
  };

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      {evalResults && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Test Return",    value: `+${(evalResults.total_return * 100).toFixed(1)}%`, color: "text-emerald-400" },
            { label: "Sortino",        value: evalResults.sortino.toFixed(2),                     color: "text-blue-400" },
            { label: "SPY Return",     value: `+${(evalResults.spy_return * 100).toFixed(1)}%`,   color: "text-violet-400" },
            { label: "Overrides",      value: String(evalResults.n_override),                     color: "text-amber-400" },
          ].map(m => (
            <div key={m.label} className="card">
              <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{m.label}</p>
              <p className={`text-xl font-bold ${m.color}`}>{m.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Action distribution */}
      {evalResults && (
        <div className="card">
          <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
            Final Action Distribution (2024 test set)
          </p>
          <div className="flex gap-4 flex-wrap">
            {Object.entries(evalResults.action_distribution).map(([name, days]) => {
              const total = Object.values(evalResults.action_distribution).reduce((s, v) => s + v, 0);
              const pct = total > 0 ? (days / total) * 100 : 0;
              const colors: Record<string, string> = { XLK: "#3b82f6", XLF: "#10b981", XLV: "#f59e0b", CASH: "#6b7280" };
              return (
                <div key={name} className="flex items-center gap-2 min-w-[120px]">
                  <div className="w-2 h-2 rounded-sm" style={{ background: colors[name] ?? "#888" }} />
                  <span className="text-xs text-slate-400">{name}</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden min-w-[60px]">
                    <div className="h-full rounded-full" style={{ width: `${pct}%`, background: colors[name] ?? "#888" }} />
                  </div>
                  <span className="text-xs font-mono text-slate-300">{days}d</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Training reward curve */}
      <div className="card">
        <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
          Training Reward per Episode (smoothed 30-ep window)
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rewardData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
            <XAxis dataKey="episode" {...axisProps} tickFormatter={v => `${v}`} />
            <YAxis {...axisProps} width={48} />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: "#94a3b8" }}
              itemStyle={{ color: "#60a5fa" }}
              formatter={(v: number) => [v.toFixed(2), "Reward"]}
              labelFormatter={(l) => `Episode ${l}`}
            />
            <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Reward" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Loss curve */}
        <div className="card">
          <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
            Bellman Loss (smoothed)
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossData} {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
              <XAxis dataKey="episode" {...axisProps} tickFormatter={v => `${v}`} />
              <YAxis {...axisProps} width={48} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: "#94a3b8" }}
                itemStyle={{ color: "#ef4444" }}
                formatter={(v: number) => [v.toFixed(4), "Loss"]}
                labelFormatter={(l) => `Episode ${l}`}
              />
              <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Epsilon decay */}
        <div className="card">
          <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">
            Epsilon (exploration) Decay
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={epsilonData} {...chartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
              <XAxis dataKey="episode" {...axisProps} tickFormatter={v => `${v}`} />
              <YAxis {...axisProps} width={36} domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: "#94a3b8" }}
                itemStyle={{ color: "#f59e0b" }}
                formatter={(v: number) => [v.toFixed(3), "Epsilon"]}
                labelFormatter={(l) => `Episode ${l}`}
              />
              <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Training info */}
      <div className="card">
        <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">Architecture</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
          {[
            ["Network",      "9 → 128 → 128 → 4"],
            ["Episodes",     `${data.ep_rewards.length.toLocaleString()}`],
            ["Batch size",   "64"],
            ["Buffer",       "10,000"],
            ["LR",           "0.001"],
            ["Gamma",        "0.99"],
            ["Grad clip",    "1.0"],
            ["ε final",      "0.05"],
          ].map(([k, v]) => (
            <div key={k}>
              <p className="text-slate-600 text-[10px] mb-0.5">{k}</p>
              <p className="text-slate-300">{v}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
