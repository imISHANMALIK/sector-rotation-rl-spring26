"use client";

import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  date: string;
  agent: number;
  spy: number;
  override: boolean;
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-2 font-mono">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: entry.color }} />
          <span className="text-slate-300">{entry.name}:</span>
          <span className="font-bold tabular-nums" style={{ color: entry.color }}>
            {((entry.value - 1) * 100).toFixed(2)}%
          </span>
        </p>
      ))}
    </div>
  );
};

const formatPct = (v: number) => `${((v - 1) * 100).toFixed(0)}%`;

const formatDate = (d: string) => {
  if (!d) return "";
  const [, month, day] = d.split("-");
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${months[parseInt(month) - 1]} ${parseInt(day)}`;
};

export default function EquityCurve({ data }: { data: DataPoint[] }) {
  if (data.length === 0) {
    return (
      <div className="h-[320px] flex items-center justify-center text-slate-600 text-sm">
        Press Run to watch the agent trade
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="agentGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="spyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#8b5cf6" stopOpacity={0.1} />
            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />

        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 10, fontFamily: "monospace" }}
          tickFormatter={formatDate}
          interval={Math.floor(data.length / 6)}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 10, fontFamily: "monospace" }}
          tickFormatter={formatPct}
          domain={["auto", "auto"]}
          width={52}
        />

        <Tooltip content={<CustomTooltip />} />

        <ReferenceLine y={1} stroke="#334155" strokeDasharray="4 2" strokeWidth={1} />

        {/* SPY fill */}
        <Area
          type="monotone"
          dataKey="spy"
          fill="url(#spyGrad)"
          stroke="none"
          isAnimationActive={false}
          legendType="none"
        />
        {/* Agent fill */}
        <Area
          type="monotone"
          dataKey="agent"
          fill="url(#agentGrad)"
          stroke="none"
          isAnimationActive={false}
          legendType="none"
        />
        {/* SPY line */}
        <Line
          type="monotone"
          dataKey="spy"
          stroke="#8b5cf6"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="5 3"
          name="SPY"
          isAnimationActive={false}
        />
        {/* Agent line */}
        <Line
          type="monotone"
          dataKey="agent"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="Agent"
          isAnimationActive={false}
        />

        <Legend
          wrapperStyle={{ fontSize: 11, color: "#94a3b8", paddingTop: 8 }}
          iconType="line"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
