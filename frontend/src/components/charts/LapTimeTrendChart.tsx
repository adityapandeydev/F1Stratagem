import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { Lap } from "../../types";

interface LapTimeTrendChartProps {
  laps: Lap[];
}

export default function LapTimeTrendChart({ laps }: LapTimeTrendChartProps) {
  const option = useMemo(() => {
    // Group laps by driver_id
    const driverLaps = new Map<number, Lap[]>();
    laps.forEach((l) => {
      if (l.is_deleted || !l.lap_time_ms) return;
      if (!driverLaps.has(l.driver_id)) {
        driverLaps.set(l.driver_id, []);
      }
      driverLaps.get(l.driver_id)!.push(l);
    });

    const series: any[] = [];
    const colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];
    let cIdx = 0;

    driverLaps.forEach((dLaps, driverId) => {
      // Sort by lap number
      dLaps.sort((a, b) => a.lap_number - b.lap_number);
      const color = colors[cIdx % colors.length];
      cIdx++;

      series.push({
        name: `Driver ${driverId}`,
        type: "line",
        data: dLaps.map((l) => [l.lap_number, (l.lap_time_ms! / 1000)]),
        showSymbol: false,
        smooth: true,
        lineStyle: { width: 1.5, color },
        itemStyle: { color },
      });
    });

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(18, 18, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.1)",
        textStyle: { color: "#ededed", fontFamily: "Inter, sans-serif", fontSize: 12 },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          let html = `<div style="font-weight: 600; margin-bottom: 6px; font-family: monospace;">Lap ${params[0].value[0]}</div>`;
          params.forEach((item: any) => {
            const timeSec = item.value[1];
            const m = Math.floor(timeSec / 60);
            const s = (timeSec % 60).toFixed(3);
            html += `<div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 11px;">
              <span style="color: ${item.color};">${item.seriesName}</span>
              <span style="font-family: monospace; font-weight: 600;">${m}:${s.padStart(6, "0")}</span>
            </div>`;
          });
          return html;
        },
      },
      grid: { left: "55px", right: "20px", top: "30px", bottom: "35px" },
      xAxis: {
        type: "value",
        name: "Lap",
        nameTextStyle: { color: "#71717a", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
        axisLabel: { color: "#71717a", fontSize: 10 },
      },
      yAxis: {
        type: "value",
        name: "Lap Time (s)",
        nameTextStyle: { color: "#71717a", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
        axisLabel: {
          color: "#71717a",
          fontSize: 10,
          formatter: (v: number) => {
            const m = Math.floor(v / 60);
            const s = (v % 60).toFixed(0);
            return `${m}:${s.padStart(2, "0")}`;
          },
        },
      },
      series,
    };
  }, [laps]);

  if (laps.length === 0) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-6 shadow-2xl">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-text-primary">Lap Pace Evolution</h3>
        <p className="text-xs text-text-secondary mt-0.5">
          Driver lap time progression, tire degradation drop-off, and pace trends
        </p>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
}
