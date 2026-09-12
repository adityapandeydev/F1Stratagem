import React, { useMemo, useRef, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import { TelemetryPoint, TelemetryDriverMeta } from "../../types";

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
  timeDelta = [],
  currentDistance,
  onDistanceHover,
}: TelemetryComparisonChartProps) {
  const chartRef = useRef<any>(null);

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

  const option = useMemo(() => {
    if (primaryTel.length === 0) return {};

    const distances = primaryTel.map((p) => p.d);

    // Build series dynamically for each driver
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

    const deltaSeries = [
      {
        name: `Delta to ${drivers[0]?.abbreviation || "Ref"}`,
        type: "line",
        xAxisIndex: 3,
        yAxisIndex: 3,
        showSymbol: false,
        lineStyle: { color: drivers[1]?.color || "#ef4444", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: `${drivers[1]?.color || "#ef4444"}33` },
              { offset: 1, color: "transparent" },
            ],
          },
        },
        data: timeDelta.length > 0 ? timeDelta : distances.map(() => 0),
      },
    ];

    return {
      backgroundColor: "transparent",
      animation: false,
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          lineStyle: { color: "rgba(255, 255, 255, 0.4)", width: 1, type: "dashed" },
        },
        backgroundColor: "rgba(16, 16, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#ededed", fontFamily: "Inter, sans-serif", fontSize: 11 },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          const d = params[0].axisValue;
          let html = `<div style="font-weight: 600; margin-bottom: 6px; font-family: monospace; color: #a1a1aa;">Track Dist: ${Math.round(d)}m</div>`;
          params.forEach((item: any) => {
            html += `<div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 2px 0;">
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
        { left: "55px", right: "20px", top: "25px", height: "180px" }, // Speed
        { left: "55px", right: "20px", top: "235px", height: "100px" }, // Throttle & Brake
        { left: "55px", right: "20px", top: "360px", height: "80px" },  // Gear
        { left: "55px", right: "20px", top: "465px", height: "70px" },  // Delta
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
        // Grid 3: Delta
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
        ...speedSeries,
        ...throttleSeries,
        ...brakeSeries,
        ...gearSeries,
        ...deltaSeries,
      ],
    };
  }, [drivers, telemetryStreams, timeDelta, primaryTel]);

  // Synchronize cursor when scrubbing from external slider
  useEffect(() => {
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

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-5 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Telemetry Trace Analysis
          </h3>
        </div>
        <div className="flex items-center gap-3">
          {drivers.map((d) => (
            <div key={d.abbreviation} className="flex items-center gap-1.5 text-xs font-mono font-medium">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
              <span style={{ color: d.color }}>{d.abbreviation}</span>
              {d.lapTimeMs && (
                <span className="text-neutral-400 text-[10px]">
                  {((d.lapTimeMs % 60000) / 1000).toFixed(3)}s
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: "560px", width: "100%" }}
        onEvents={{ updateAxisPointer: onChartHover }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
