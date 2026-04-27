"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  date: string;
  z_xlk: number;
  z_xlf: number;
  z_xlv: number;
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  const allHigh = payload.every(p => p.value > 2.5);
  return (
    <div className={`border rounded-lg p-2.5 text-xs shadow-xl ${allHigh ? "bg-red-950 border-red-700" : "bg-slate-900 border-slate-700"}`}>
      <p className="text-slate-400 font-mono mb-1">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="flex items-center gap-2 mb-0.5">
          <span className="w-2 h-2 rounded-full" style={{ background: entry.color }} />
          <span className="text-slate-300">{entry.name}:</span>
          <span className={`font-bold tabular-nums ${Math.abs(entry.value) > 2.5 ? "text-red-400" : ""}`} style={Math.abs(entry.value) <= 2.5 ? { color: entry.color } : {}}>
            {entry.value.toFixed(2)}
          </span>
        </p>
      ))}
      {allHigh && <p className="text-red-400 text-[10px] mt-1 font-bold">⚠ Override zone</p>}
    </div>
  );
};

const formatDate = (d: string) => {
  if (!d) return "";
  const [, month] = d.split("-");
  const months = ["J","F","M","A","M","J","J","A","S","O","N","D"];
  return months[parseInt(month) - 1];
};

export default function IVZScoreChart({ data }: { data: DataPoint[] }) {
  if (data.length === 0) {
    return (
      <div className="h-[210px] flex items-center justify-center text-slate-600 text-sm">
        No data yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={210}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
          tickFormatter={formatDate}
          interval={Math.floor(data.length / 5)}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
          tickFormatter={v => v.toFixed(1)}
          domain={[-3.5, 3.5]}
          width={32}
        />
        <Tooltip content={<CustomTooltip />} />

        {/* Override threshold lines */}
        <ReferenceLine y={2.5}  stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} label={{ value: "2.5σ", position: "right", fill: "#ef4444", fontSize: 9 }} />
        <ReferenceLine y={-2.5} stroke="#22c55e" strokeDasharray="4 2" strokeWidth={1} />
        <ReferenceLine y={0}    stroke="#334155" strokeDasharray="2 2" strokeWidth={1} />

        <Line type="monotone" dataKey="z_xlk" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Z XLK" isAnimationActive={false} />
        <Line type="monotone" dataKey="z_xlf" stroke="#10b981" strokeWidth={1.5} dot={false} name="Z XLF" isAnimationActive={false} />
        <Line type="monotone" dataKey="z_xlv" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Z XLV" isAnimationActive={false} />

        <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} iconType="line" />
      </LineChart>
    </ResponsiveContainer>
  );
}
