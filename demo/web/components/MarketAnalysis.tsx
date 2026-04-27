"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface MarketRow {
  date: string;
  iv_xlk: number;
  iv_xlf: number;
  iv_xlv: number;
  zscore_xlk: number;
  zscore_xlf: number;
  zscore_xlv: number;
  realvol_xlk: number;
  realvol_xlf: number;
  realvol_xlv: number;
  ret_xlk: number;
  ret_xlf: number;
  ret_xlv: number;
  ret_spy: number;
  rf_daily: number;
}

function subsample<T>(arr: T[], n = 500): T[] {
  if (arr.length <= n) return arr;
  const step = Math.ceil(arr.length / n);
  return arr.filter((_, i) => i % step === 0);
}

function makeHistogram(vals: number[], bins = 40): { x: string; count: number }[] {
  const filtered = vals.filter(v => isFinite(v) && !isNaN(v));
  if (!filtered.length) return [];
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  const bw = (max - min) / bins;
  const buckets = Array.from({ length: bins }, (_, i) => ({
    x: ((min + (i + 0.5) * bw) * 100).toFixed(2),
    count: 0,
  }));
  filtered.forEach(v => {
    const b = Math.min(Math.floor((v - min) / bw), bins - 1);
    buckets[b].count++;
  });
  return buckets;
}

const axisStyle = { fill: "#475569", fontSize: 10 };

const tooltipStyle = {
  contentStyle: { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 },
  labelStyle: { color: "#94a3b8" },
};

const formatDate = (d: string) => {
  if (!d) return "";
  const [year, month] = d.split("-");
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${months[parseInt(month) - 1]} '${year.slice(2)}`;
};

export default function MarketAnalysis({ data }: { data: Record<string, unknown>[] }) {
  const rows = data as unknown as MarketRow[];

  const subRows  = useMemo(() => subsample(rows, 500), [rows]);
  const testRows = useMemo(() => subRows.filter(r => r.date >= "2024-01-01"), [subRows]);

  const ivData = useMemo(() => subRows.map(r => ({
    date: r.date,
    xlk: r.iv_xlk,
    xlf: r.iv_xlf,
    xlv: r.iv_xlv,
  })), [subRows]);

  const zData = useMemo(() => subRows.map(r => ({
    date:  r.date,
    z_xlk: r.zscore_xlk,
    z_xlf: r.zscore_xlf,
    z_xlv: r.zscore_xlv,
  })), [subRows]);

  const rvData = useMemo(() => subRows.map(r => ({
    date:   r.date,
    rv_xlk: r.realvol_xlk,
    rv_xlf: r.realvol_xlf,
    rv_xlv: r.realvol_xlv,
    iv_xlk: r.iv_xlk,
    iv_xlf: r.iv_xlf,
    iv_xlv: r.iv_xlv,
  })), [subRows]);

  const retHistXlk  = useMemo(() => makeHistogram(rows.map(r => r.ret_xlk)), [rows]);
  const retHistXlf  = useMemo(() => makeHistogram(rows.map(r => r.ret_xlf)), [rows]);
  const retHistXlv  = useMemo(() => makeHistogram(rows.map(r => r.ret_xlv)), [rows]);
  const retHistSpy  = useMemo(() => makeHistogram(rows.map(r => r.ret_spy)), [rows]);

  const cumRetData = useMemo(() => {
    let xlk = 1, xlf = 1, xlv = 1, spy = 1;
    return rows.map(r => {
      xlk *= Math.exp(r.ret_xlk || 0);
      xlf *= Math.exp(r.ret_xlf || 0);
      xlv *= Math.exp(r.ret_xlv || 0);
      spy *= Math.exp(r.ret_spy  || 0);
      return { date: r.date, xlk, xlf, xlv, spy };
    });
  }, [rows]);
  const cumSub = useMemo(() => subsample(cumRetData, 500), [cumRetData]);

  if (rows.length === 0) {
    return <div className="flex items-center justify-center h-64 text-slate-600">Loading market data...</div>;
  }

  const sectionTitle = (t: string) => (
    <p className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold mb-3">{t}</p>
  );

  return (
    <div className="space-y-4">
      {/* Dataset info */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
        {[
          { label: "Total Days",  value: rows.length.toString() },
          { label: "Train Days",  value: rows.filter(r => r.date < "2024-01-01").length.toString() },
          { label: "Test Days",   value: rows.filter(r => r.date >= "2024-01-01").length.toString() },
          { label: "Coverage",    value: `${rows[0]?.date?.slice(0,7)} → ${rows[rows.length-1]?.date?.slice(0,7)}` },
          { label: "Features",    value: "9-dim state" },
          { label: "Actions",     value: "XLK / XLF / XLV / CASH" },
        ].map(m => (
          <div key={m.label} className="card">
            <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{m.label}</p>
            <p className="text-xs font-mono text-slate-200">{m.value}</p>
          </div>
        ))}
      </div>

      {/* Implied Volatility */}
      <div className="card">
        {sectionTitle("Implied Volatility — All Sectors (2020-2024)")}
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={ivData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} tick={axisStyle} tickFormatter={formatDate} interval={Math.floor(ivData.length / 6)} />
            <YAxis tickLine={false} axisLine={false} tick={axisStyle} width={36} tickFormatter={v => v.toFixed(2)} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [v.toFixed(3), ""]} labelFormatter={formatDate} />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
            <Line type="monotone" dataKey="xlk" stroke="#3b82f6" strokeWidth={1.2} dot={false} name="XLK IV" isAnimationActive={false} />
            <Line type="monotone" dataKey="xlf" stroke="#10b981" strokeWidth={1.2} dot={false} name="XLF IV" isAnimationActive={false} />
            <Line type="monotone" dataKey="xlv" stroke="#f59e0b" strokeWidth={1.2} dot={false} name="XLV IV" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Z-Score Timeline */}
      <div className="card">
        {sectionTitle("IV Z-Scores (60-day rolling) — Override Zone > 2.5")}
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={zData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} tick={axisStyle} tickFormatter={formatDate} interval={Math.floor(zData.length / 6)} />
            <YAxis tickLine={false} axisLine={false} tick={axisStyle} width={36} domain={[-4, 6]} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [v.toFixed(2), ""]} labelFormatter={formatDate} />
            <ReferenceLine y={2.5} stroke="#ef4444" strokeDasharray="4 2" label={{ value: "Override 2.5σ", position: "insideTopRight", fill: "#ef4444", fontSize: 9 }} />
            <ReferenceLine y={0}   stroke="#334155" strokeDasharray="2 2" />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
            <Line type="monotone" dataKey="z_xlk" stroke="#3b82f6" strokeWidth={1.2} dot={false} name="Z XLK" isAnimationActive={false} />
            <Line type="monotone" dataKey="z_xlf" stroke="#10b981" strokeWidth={1.2} dot={false} name="Z XLF" isAnimationActive={false} />
            <Line type="monotone" dataKey="z_xlv" stroke="#f59e0b" strokeWidth={1.2} dot={false} name="Z XLV" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cumulative Returns */}
      <div className="card">
        {sectionTitle("Buy-and-Hold Cumulative Returns (2020-2024)")}
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={cumSub} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} tick={axisStyle} tickFormatter={formatDate} interval={Math.floor(cumSub.length / 6)} />
            <YAxis tickLine={false} axisLine={false} tick={axisStyle} width={44} tickFormatter={v => `${((v-1)*100).toFixed(0)}%`} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [`${((v-1)*100).toFixed(1)}%`, ""]} labelFormatter={formatDate} />
            <ReferenceLine y={1} stroke="#334155" strokeDasharray="2 2" />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
            <Line type="monotone" dataKey="spy" stroke="#8b5cf6" strokeWidth={2}   dot={false} name="SPY"    isAnimationActive={false} />
            <Line type="monotone" dataKey="xlk" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="XLK"    isAnimationActive={false} />
            <Line type="monotone" dataKey="xlf" stroke="#10b981" strokeWidth={1.5} dot={false} name="XLF"    isAnimationActive={false} />
            <Line type="monotone" dataKey="xlv" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="XLV"    isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Realized vs Implied vol */}
      <div className="card">
        {sectionTitle("Realized vs Implied Volatility (Vol Risk Premium)")}
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rvData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} tick={axisStyle} tickFormatter={formatDate} interval={Math.floor(rvData.length / 6)} />
            <YAxis tickLine={false} axisLine={false} tick={axisStyle} width={36} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [v.toFixed(4), ""]} labelFormatter={formatDate} />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
            <Line type="monotone" dataKey="iv_xlk" stroke="#3b82f6" strokeWidth={1.2} dot={false} name="IV XLK"  isAnimationActive={false} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="rv_xlk" stroke="#60a5fa" strokeWidth={1}   dot={false} name="RV XLK"  isAnimationActive={false} />
            <Line type="monotone" dataKey="iv_xlv" stroke="#f59e0b" strokeWidth={1.2} dot={false} name="IV XLV"  isAnimationActive={false} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="rv_xlv" stroke="#fcd34d" strokeWidth={1}   dot={false} name="RV XLV"  isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Return distributions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { title: "XLK Daily Returns", data: retHistXlk, color: "#3b82f6" },
          { title: "XLF Daily Returns", data: retHistXlf, color: "#10b981" },
          { title: "XLV Daily Returns", data: retHistXlv, color: "#f59e0b" },
          { title: "SPY Daily Returns", data: retHistSpy,  color: "#8b5cf6" },
        ].map(({ title, data: hist, color }) => (
          <div key={title} className="card">
            {sectionTitle(title + " Distribution (2020-2024)")}
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={hist} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#162740" vertical={false} />
                <XAxis dataKey="x" tickLine={false} axisLine={false} tick={{ fill: "#475569", fontSize: 8 }} interval={7} tickFormatter={v => `${v}%`} />
                <YAxis tickLine={false} axisLine={false} tick={{ fill: "#475569", fontSize: 9 }} width={28} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: "#94a3b8" }}
                  formatter={(v: number) => [v, "days"]}
                  labelFormatter={v => `Return: ${v}%`}
                />
                <Bar dataKey="count" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                  {hist.map((entry, i) => (
                    <Cell key={i} fill={parseFloat(entry.x) < 0 ? "#ef4444" : color} fillOpacity={0.7} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  );
}
