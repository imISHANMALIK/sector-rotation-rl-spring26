"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

interface PieData {
  name: string;
  value: number;
  color: string;
}

const CustomTooltip = ({ active, payload }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; payload: { color: string } }>;
}) => {
  if (!active || !payload?.length) return null;
  const total = payload[0].value;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs shadow-xl">
      <p style={{ color: payload[0].payload.color }} className="font-bold">{payload[0].name}</p>
      <p className="text-slate-300">{total} days</p>
    </div>
  );
};

const RADIAN = Math.PI / 180;
const renderLabel = ({
  cx, cy, midAngle, innerRadius, outerRadius, name, percent,
}: {
  cx: number; cy: number; midAngle: number;
  innerRadius: number; outerRadius: number;
  name: string; percent: number;
}) => {
  if (percent < 0.06) return null;
  const r = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + r * Math.cos(-midAngle * RADIAN);
  const y = cy + r * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={10} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export default function SectorPie({
  data,
  currentAction,
}: {
  data: PieData[];
  currentAction?: string;
}) {
  if (data.length === 0) {
    return (
      <div className="h-[210px] flex items-center justify-center text-slate-600 text-sm">
        No data yet
      </div>
    );
  }

  const total = data.reduce((s, d) => s + d.value, 0);
  const active = data.find(d => d.name === currentAction);

  return (
    <div className="flex flex-col items-center gap-2">
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={48}
            outerRadius={80}
            dataKey="value"
            labelLine={false}
            label={renderLabel}
            isAnimationActive={false}
          >
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.color}
                opacity={!currentAction || entry.name === currentAction ? 1 : 0.45}
                stroke={entry.name === currentAction ? entry.color : "transparent"}
                strokeWidth={entry.name === currentAction ? 2 : 0}
              />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-x-3 gap-y-1">
        {data.map(d => (
          <div key={d.name} className="flex items-center gap-1 text-[10px]">
            <span className="w-2 h-2 rounded-sm inline-block" style={{ background: d.color }} />
            <span className="text-slate-400">{d.name}</span>
            <span className="text-slate-500 tabular-nums">
              {total > 0 ? `${((d.value / total) * 100).toFixed(0)}%` : "0%"}
            </span>
          </div>
        ))}
      </div>

      {active && (
        <p className="text-[11px] font-semibold" style={{ color: active.color }}>
          Current: {active.name}
        </p>
      )}
    </div>
  );
}
