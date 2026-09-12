import React, { useMemo, useRef, useState, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import { RotateCcw } from "lucide-react";
import { TelemetryPoint } from "../../types";

interface DriverMeta {
  abbreviation: string;
  name?: string;
  color: string;
  lapNumber?: number;
  lapTimeMs?: number;
}

interface TelemetryComparisonChartProps {
  drivers?: DriverMeta[];
  telemetryStreams?: TelemetryPoint[][];
  timeDelta?: number[];
  currentDistance?: number;
  onDistanceHover?: (dist: number) => void;
  // Legacy props
  driver1?: DriverMeta;
  driver2?: DriverMeta;
  telemetry1?: TelemetryPoint[];
  telemetry2?: TelemetryPoint[];
}

export default function TelemetryComparisonChart({
  drivers: driversProp,
  telemetryStreams: streamsProp,
  driver1,
  driver2,
  telemetry1,
  telemetry2,
  currentDistance,
  onDistanceHover,
}: TelemetryComparisonChartProps) {
  const chartRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isZoomed, setIsZoomed] = useState(false);

  // Trap mouse wheel over chart container to prevent outer page scroll/zoom
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
    };

    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", handleWheel);
    };
  }, []);

  // Normalize drivers and streams
  const drivers: DriverMeta[] = useMemo(() => {
    if (driversProp && driversProp.length > 0) return driversProp;
    const res: DriverMeta[] = [];
    if (driver1) res.push(driver1);
    if (driver2) res.push(driver2);
    return res;
  }, [driversProp, driver1, driver2]);

  const telemetryStreams: TelemetryPoint[][] = useMemo(() => {
    if (streamsProp && streamsProp.length > 0) return streamsProp;
    const res: TelemetryPoint[][] = [];
    if (telemetry1) res.push(telemetry1);
    if (telemetry2) res.push(telemetry2);
    return res;
  }, [streamsProp, telemetry1, telemetry2]);

  const primaryTel = telemetryStreams[0] || [];

  // Active values at currentDistance for the fixed HUD
  const activeMetrics = useMemo(() => {
    if (primaryTel.length === 0) return null;
    const dist = currentDistance !== undefined ? currentDistance : 0;
    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < primaryTel.length; i++) {
      const diff = Math.abs(primaryTel[i].d - dist);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }

    return {
      distance: Math.round(primaryTel[closestIdx]?.d || dist),
      drivers: drivers.map((drv, idx) => {
        const pt = telemetryStreams[idx]?.[closestIdx] || primaryTel[closestIdx];
        return {
          ...drv,
          speed: Math.round(pt?.spd || 0),
          throttle: Math.round(pt?.thr || 0),
          brake: Math.round((pt?.brk || 0) * 100),
          gear: pt?.gear || 7,
        };
      }),
    };
  }, [primaryTel, currentDistance, drivers, telemetryStreams]);

  const option = useMemo(() => {
    if (primaryTel.length === 0) return {};

    const distances = primaryTel.map((p) => p.d);

    // Build series dynamically for each driver (Speed, Throttle, Brake, Gear)
    const speedSeries = drivers.map((drv, idx) => {
      const tel = telemetryStreams[idx] || [];
      return {
        name: `${drv.abbreviation} Speed`,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { color: drv.color || "#ffffff", width: 2.2 },
        itemStyle: { color: drv.color || "#ffffff" },
        data: tel.map((p) => p.spd),
      };
    });

    const throttleSeries = drivers.map((drv, idx) => {
      const tel = telemetryStreams[idx] || [];
      return {
        name: `${drv.abbreviation} Thr`,
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: drv.color || "#ffffff", width: 1.8 },
        itemStyle: { color: drv.color || "#ffffff" },
        data: tel.map((p) => p.thr),
      };
    });

    const brakeSeries = drivers.map((drv, idx) => {
      const tel = telemetryStreams[idx] || [];
      return {
        name: `${drv.abbreviation} Brk`,
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: drv.color || "#ffffff", width: 1.5, type: "dashed" },
        itemStyle: { color: drv.color || "#ffffff" },
        data: tel.map((p) => p.brk * 100),
      };
    });

    const gearSeries = drivers.map((drv, idx) => {
      const tel = telemetryStreams[idx] || [];
      return {
        name: `${drv.abbreviation} Gear`,
        type: "line",
        step: "end",
        xAxisIndex: 2,
        yAxisIndex: 2,
        showSymbol: false,
        lineStyle: { color: drv.color || "#ffffff", width: 1.6 },
        itemStyle: { color: drv.color || "#ffffff" },
        data: tel.map((p) => p.gear),
      };
    });

    return {
      backgroundColor: "transparent",
      animation: false,
      tooltip: {
        trigger: "axis",
        showContent: false, // Hides floating popup HTML box while preserving canvas vertical tracking line and dots!
        axisPointer: {
          type: "cross",
          lineStyle: { color: "rgba(255, 255, 255, 0.45)", width: 1.5, type: "dashed" },
          label: { show: false },
        },
      },
      axisPointer: {
        link: [{ xAxisIndex: "all" }],
      },
      grid: [
        { left: "55px", right: "25px", top: "20px", height: "195px" }, // 1. Speed (km/h)
        { left: "55px", right: "25px", top: "250px", height: "125px" }, // 2. Pedals (%)
        { left: "55px", right: "25px", top: "410px", height: "90px" },  // 3. Gear (1-8)
      ],
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1, 2],
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
          moveOnMouseWheel: false,
        },
        {
          type: "slider",
          xAxisIndex: [0, 1, 2],
          bottom: "6px",
          height: "18px",
          borderColor: "rgba(255, 255, 255, 0.08)",
          backgroundColor: "#0b0b10",
          fillerColor: "rgba(239, 68, 68, 0.22)",
          dataBackground: {
            lineStyle: { color: "#3f3f46" },
            areaStyle: { color: "rgba(255,255,255,0.04)" },
          },
          selectedDataBackground: {
            lineStyle: { color: "#ef4444" },
            areaStyle: { color: "rgba(239, 68, 68, 0.2)" },
          },
          handleStyle: {
            color: "#ef4444",
            borderColor: "#ffffff",
            borderWidth: 1,
          },
          textStyle: {
            color: "#a1a1aa",
            fontSize: 9,
            fontFamily: "monospace",
          },
        },
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
      ],
      series: [
        ...speedSeries,
        ...throttleSeries,
        ...brakeSeries,
        ...gearSeries,
      ],
    };
  }, [drivers, telemetryStreams, primaryTel]);

  // Synchronize cursor and tracer dots during scrubbing & playback
  React.useEffect(() => {
    if (!chartRef.current || currentDistance === undefined || primaryTel.length === 0) return;
    const echartInstance = chartRef.current.getEchartsInstance();
    if (!echartInstance) return;

    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < primaryTel.length; i++) {
      const diff = Math.abs(primaryTel[i].d - currentDistance);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }

    echartInstance.dispatchAction({
      type: "showTip",
      seriesIndex: 0,
      dataIndex: closestIdx,
    });
  }, [currentDistance, primaryTel]);

  const onChartHover = (params: any) => {
    if (onDistanceHover && params && params.dataIndex !== undefined) {
      const pt = primaryTel[params.dataIndex];
      if (pt) onDistanceHover(pt.d);
    }
  };

  const handleResetZoom = () => {
    if (!chartRef.current) return;
    const inst = chartRef.current.getEchartsInstance();
    if (!inst) return;
    inst.dispatchAction({
      type: "dataZoom",
      start: 0,
      end: 100,
    });
    setIsZoomed(false);
  };

  return (
    <div
      ref={containerRef}
      className="rounded-2xl border border-white/[0.08] bg-[#070709] backdrop-blur-xl p-5 shadow-2xl space-y-4"
    >
      {/* Header Container with Title on Line 1 and Responsive HUD Bar on Line 2 */}
      <div className="space-y-3 pb-3 border-b border-white/[0.06]">
        {/* Line 1: Title, Subtitle, and Distance Badge */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-white tracking-wide">
                Telemetry Trace Synchronizer
              </h3>
              <p className="text-[11px] text-neutral-400 font-mono">
                Speed (km/h) · Pedals (%) · Transmission Gear · Mouse Wheel to Zoom
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 shrink-0 self-start sm:self-auto">
            {isZoomed && (
              <button
                onClick={handleResetZoom}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-400 hover:text-white text-xs font-mono font-semibold transition-all shadow-sm cursor-pointer"
                title="Reset to Full Lap"
              >
                <RotateCcw className="w-3 h-3" />
                Reset Zoom
              </button>
            )}

            {activeMetrics && (
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider hidden md:inline">
                  Track Distance
                </span>
                <div className="px-3 py-1.5 rounded-xl bg-[#101014] border border-white/[0.08] text-xs font-mono font-bold text-neutral-200 tabular-nums shadow-sm">
                  {activeMetrics.distance.toLocaleString()}m
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Line 2: Responsive Live Driver HUD Bar (2x2 on laptops/smaller screens, 4x1 on desktop, NO scrollbars) */}
        {activeMetrics && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 w-full pt-1">
            {activeMetrics.drivers.map((d) => (
              <div
                key={d.abbreviation}
                className="flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-[#101014] border border-white/[0.08] text-xs font-mono shadow-sm"
              >
                <div className="flex items-center gap-1.5 min-w-0 shrink-0">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: d.color }} />
                  <span className="font-extrabold" style={{ color: d.color }}>{d.abbreviation}</span>
                </div>
                
                {/* Speed */}
                <div className="text-white font-bold tabular-nums text-right">
                  {d.speed} <span className="text-[10px] text-neutral-400 font-normal">km/h</span>
                </div>
                
                <span className="text-white/15">|</span>
                
                {/* Throttle % */}
                <div className={`tabular-nums text-[11px] font-semibold text-right ${d.throttle > 0 ? "text-emerald-400" : "text-neutral-500"}`}>
                  {d.throttle}% <span className="text-[9px] text-neutral-400 font-normal">Thr</span>
                </div>
                
                {/* Brake % */}
                <div className={`tabular-nums text-[11px] font-semibold text-right ${d.brake > 0 ? "text-red-400 font-bold" : "text-neutral-600"}`}>
                  {d.brake}% <span className="text-[9px] text-neutral-400 font-normal">Brk</span>
                </div>
                
                <span className="text-white/15">|</span>
                
                {/* Gear */}
                <div className="text-amber-300 font-bold tabular-nums text-center">
                  G{d.gear}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Synchronized 3-Subplot ECharts Canvas */}
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: "590px", width: "100%" }}
        onEvents={{
          updateAxisPointer: onChartHover,
          datazoom: () => setIsZoomed(true),
        }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}

