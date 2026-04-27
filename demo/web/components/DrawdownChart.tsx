"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  date: string;
  drawdown: number;
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs shadow-xl">
      <p className="text-slate-400 font-mono mb-1">{label}</p>
      <p className="text-red-400 font-bold tabular-nums">{payload[0].value.toFixed(2)}%</p>
    </div>
  );
};

const formatDate = (d: string) => {
  if (!d) return "";
  const [, month] = d.split("-");
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return months[parseInt(month) - 1];
};

export default function DrawdownChart({ data }: { data: DataPoint[] }) {
  if (data.length === 0) {
    return (
      <div className="h-[210px] flex items-center justify-center text-slate-600 text-sm">
        No data yet
      </div>
    );
  }

  const minDD = Math.min(...data.map(d => d.drawdown));

  return (
    <ResponsiveContainer width="100%" height={210}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.35} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
          tickFormatter={formatDate}
          interval={Math.floor(data.length / 4)}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
          tickFormatter={v => `${v.toFixed(0)}%`}
          domain={[Math.floor(minDD * 1.2), 0]}
          width={40}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 2" />

        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
