import React, { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { TelemetryPoint } from "../../types";

interface DriverMeta {
  abbreviation: string;
  name?: string;
  color: string;
  lapNumber?: number;
  lapTimeMs?: number;
}

interface TelemetryComparisonChartProps {
  telemetry1: TelemetryPoint[];
  telemetry2: TelemetryPoint[];
  driver1: DriverMeta;
  driver2: DriverMeta;
  timeDelta?: number[];
  currentDistance?: number;
  onDistanceHover?: (dist: number) => void;
}

export default function TelemetryComparisonChart({
  telemetry1,
  telemetry2,
  driver1,
  driver2,
  timeDelta = [],
  currentDistance,
  onDistanceHover,
}: TelemetryComparisonChartProps) {
  const chartRef = useRef<any>(null);

  const formatLapTime = (ms?: number) => {
    if (!ms) return "--:--.---";
    const mins = Math.floor(ms / 60000);
    const secs = ((ms % 60000) / 1000).toFixed(3);
    return `${mins}:${secs.padStart(6, "0")}`;
  };

  const option = useMemo(() => {
    const distances = telemetry1.map((p) => p.d);
    const speed1 = telemetry1.map((p) => p.spd);
    const speed2 = telemetry2.map((p) => p.spd);
    const throttle1 = telemetry1.map((p) => p.thr);
    const throttle2 = telemetry2.map((p) => p.thr);
    const brake1 = telemetry1.map((p) => p.brk * 100);
    const brake2 = telemetry2.map((p) => p.brk * 100);
    const gear1 = telemetry1.map((p) => p.gear);
    const gear2 = telemetry2.map((p) => p.gear);
    const rpm1 = telemetry1.map((p) => p.rpm);
    const rpm2 = telemetry2.map((p) => p.rpm);
    const drs1 = telemetry1.map((p) => (p.drs >= 10 ? 1 : 0));
    const drs2 = telemetry2.map((p) => (p.drs >= 10 ? 1 : 0));

    return {
      backgroundColor: "transparent",
      animation: false,
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          lineStyle: { color: "rgba(255, 255, 255, 0.4)", width: 1, type: "dashed" },
        },
        backgroundColor: "rgba(18, 18, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.1)",
        textStyle: { color: "#ededed", fontFamily: "Inter, sans-serif", fontSize: 12 },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          const d = params[0].axisValue;
          let html = `<div style="font-weight: 600; margin-bottom: 6px; font-family: monospace; color: #a1a1aa;">Track Dist: ${Math.round(d)}m</div>`;
          params.forEach((item: any) => {
            html += `<div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 3px 0;">
              <span style="display: flex; align-items: center; gap: 6px; font-size: 11px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: ${item.color};"></span>
                ${item.seriesName}
              </span>
              <span style="font-family: monospace; font-weight: 600;">${item.value !== undefined ? item.value : "-"}</span>
            </div>`;
          });
          return html;
        },
      },
      axisPointer: {
        link: [{ xAxisIndex: "all" }],
      },
      grid: [
        { left: "55px", right: "20px", top: "20px", height: "180px" }, // Speed
        { left: "55px", right: "20px", top: "225px", height: "100px" }, // Throttle & Brake
        { left: "55px", right: "20px", top: "350px", height: "80px" },  // Gear
        { left: "55px", right: "20px", top: "455px", height: "70px" },  // Delta / DRS
      ],
      xAxis: [
        {
          gridIndex: 0,
          type: "category",
          data: distances,
          show: false,
          axisLine: { lineStyle: { color: "#27272a" } },
        },
        {
          gridIndex: 1,
          type: "category",
          data: distances,
          show: false,
          axisLine: { lineStyle: { color: "#27272a" } },
        },
        {
          gridIndex: 2,
          type: "category",
          data: distances,
          show: false,
          axisLine: { lineStyle: { color: "#27272a" } },
        },
        {
          gridIndex: 3,
          type: "category",
          data: distances,
          show: true,
          axisLine: { lineStyle: { color: "#27272a" } },
          axisLabel: {
            color: "#71717a",
            fontSize: 10,
            formatter: (v: any) => `${Math.round(v)}m`,
          },
        },
      ],
      yAxis: [
        // Grid 0: Speed
        {
          gridIndex: 0,
          type: "value",
          name: "Speed (km/h)",
          nameTextStyle: { color: "#71717a", fontSize: 10, align: "left" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
          axisLabel: { color: "#71717a", fontSize: 10 },
          min: 60,
          max: 360,
        },
        // Grid 1: Throttle & Brake
        {
          gridIndex: 1,
          type: "value",
          name: "Pedals (%)",
          nameTextStyle: { color: "#71717a", fontSize: 10, align: "left" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
          axisLabel: { color: "#71717a", fontSize: 10 },
          min: 0,
          max: 100,
        },
        // Grid 2: Gear
        {
          gridIndex: 2,
          type: "value",
          name: "Gear",
          nameTextStyle: { color: "#71717a", fontSize: 10, align: "left" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
          axisLabel: { color: "#71717a", fontSize: 10 },
          min: 1,
          max: 8,
          interval: 1,
        },
        // Grid 3: Delta Time (s)
        {
          gridIndex: 3,
          type: "value",
          name: "Delta (s)",
          nameTextStyle: { color: "#71717a", fontSize: 10, align: "left" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.04)" } },
          axisLabel: { color: "#71717a", fontSize: 10 },
        },
      ],
      series: [
        // Speed Panel
        {
          name: `${driver1.abbreviation} Speed`,
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: speed1,
          showSymbol: false,
          lineStyle: { width: 2, color: driver1.color || "#3b82f6" },
        },
        {
          name: `${driver2.abbreviation} Speed`,
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: speed2,
          showSymbol: false,
          lineStyle: { width: 2, color: driver2.color || "#ef4444" },
        },
        // Pedals Panel: Throttle
        {
          name: `${driver1.abbreviation} Throttle`,
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: throttle1,
          showSymbol: false,
          lineStyle: { width: 1.5, color: driver1.color || "#3b82f6" },
        },
        {
          name: `${driver2.abbreviation} Throttle`,
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: throttle2,
          showSymbol: false,
          lineStyle: { width: 1.5, color: driver2.color || "#ef4444" },
        },
        // Pedals Panel: Brake
        {
          name: `${driver1.abbreviation} Brake`,
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: brake1,
          showSymbol: false,
          step: "end",
          lineStyle: { width: 1.5, color: "#f87171", type: "dashed" },
          areaStyle: { color: "rgba(248, 113, 113, 0.1)" },
        },
        {
          name: `${driver2.abbreviation} Brake`,
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: brake2,
          showSymbol: false,
          step: "end",
          lineStyle: { width: 1.5, color: "#fb923c", type: "dashed" },
          areaStyle: { color: "rgba(251, 146, 60, 0.1)" },
        },
        // Gear Panel
        {
          name: `${driver1.abbreviation} Gear`,
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: gear1,
          showSymbol: false,
          step: "end",
          lineStyle: { width: 1.5, color: driver1.color || "#3b82f6" },
        },
        {
          name: `${driver2.abbreviation} Gear`,
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: gear2,
          showSymbol: false,
          step: "end",
          lineStyle: { width: 1.5, color: driver2.color || "#ef4444" },
        },
        // Delta Panel
        {
          name: "Time Delta",
          type: "line",
          xAxisIndex: 3,
          yAxisIndex: 3,
          data: timeDelta,
          showSymbol: false,
          lineStyle: { width: 1.5, color: "#a1a1aa" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(59, 130, 246, 0.25)" },
                { offset: 1, color: "rgba(239, 68, 68, 0.25)" },
              ],
            },
          },
        },
      ],
    };
  }, [telemetry1, telemetry2, driver1, driver2, timeDelta]);

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-6 shadow-2xl">
      {/* Drivers Header Card */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-2 border-b border-white/[0.06]">
        {/* Driver 1 */}
        <div className="flex items-center gap-3">
          <div
            className="w-3.5 h-10 rounded-full"
            style={{ backgroundColor: driver1.color || "#3b82f6" }}
          />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xl font-bold tracking-tight text-white">
                {driver1.abbreviation}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/[0.06] text-text-secondary font-mono">
                Lap {driver1.lapNumber || "Fastest"}
              </span>
            </div>
            <div className="text-xs text-text-tertiary font-mono mt-0.5">
              Time: <span className="text-text-primary font-medium">{formatLapTime(driver1.lapTimeMs)}</span>
            </div>
          </div>
        </div>

        {/* Delta Summary Pill */}
        <div className="px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
          <div className="text-[10px] uppercase font-mono tracking-wider text-text-tertiary">
            Gap (Lap Time)
          </div>
          <div className="text-sm font-mono font-bold text-emerald-400">
            {driver1.lapTimeMs && driver2.lapTimeMs
              ? `${((driver1.lapTimeMs - driver2.lapTimeMs) / 1000).toFixed(3)}s`
              : "Delta Synced"}
          </div>
        </div>

        {/* Driver 2 */}
        <div className="flex items-center gap-3 text-right">
          <div>
            <div className="flex items-center justify-end gap-2">
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/[0.06] text-text-secondary font-mono">
                Lap {driver2.lapNumber || "Fastest"}
              </span>
              <span className="font-mono text-xl font-bold tracking-tight text-white">
                {driver2.abbreviation}
              </span>
            </div>
            <div className="text-xs text-text-tertiary font-mono mt-0.5">
              Time: <span className="text-text-primary font-medium">{formatLapTime(driver2.lapTimeMs)}</span>
            </div>
          </div>
          <div
            className="w-3.5 h-10 rounded-full"
            style={{ backgroundColor: driver2.color || "#ef4444" }}
          />
        </div>
      </div>

      {/* Synchronized Multi-Panel ECharts */}
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: "550px", width: "100%" }}
        onEvents={{
          updateAxisPointer: (params: any) => {
            if (params.dataIndex !== undefined && telemetry1[params.dataIndex]) {
              const d = telemetry1[params.dataIndex].d;
              if (onDistanceHover) onDistanceHover(d);
            }
          },
        }}
      />
    </div>
  );
}
