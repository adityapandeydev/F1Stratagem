import React from "react";
import { CloudRain, Wind, Thermometer, Gauge, Droplets } from "lucide-react";
import { WeatherData } from "../../types";

interface WeatherWidgetProps {
  weather: WeatherData[];
}

export default function WeatherWidget({ weather }: WeatherWidgetProps) {
  if (weather.length === 0) {
    return null;
  }

  // Pick the latest weather sample
  const latest = weather[weather.length - 1];

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-5 shadow-2xl">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider flex items-center gap-1.5">
          <Thermometer className="w-3.5 h-3.5 text-rose-400" />
          Track & Ambient Conditions
        </span>
        {latest.rainfall ? (
          <span className="flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400">
            <CloudRain className="w-3 h-3" /> Wet Track
          </span>
        ) : (
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400">
            Dry Track
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Track Temp */}
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.04]">
          <div className="text-[10px] uppercase font-mono text-text-tertiary">Track Temp</div>
          <div className="text-lg font-mono font-bold text-text-primary mt-0.5">
            {latest.track_temp !== undefined && latest.track_temp !== null ? `${latest.track_temp.toFixed(1)}°C` : "-"}
          </div>
        </div>

        {/* Air Temp */}
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.04]">
          <div className="text-[10px] uppercase font-mono text-text-tertiary">Air Temp</div>
          <div className="text-lg font-mono font-bold text-text-primary mt-0.5">
            {latest.air_temp !== undefined && latest.air_temp !== null ? `${latest.air_temp.toFixed(1)}°C` : "-"}
          </div>
        </div>

        {/* Humidity */}
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.04]">
          <div className="text-[10px] uppercase font-mono text-text-tertiary flex items-center gap-1">
            <Droplets className="w-3 h-3" /> Humidity
          </div>
          <div className="text-lg font-mono font-bold text-text-primary mt-0.5">
            {latest.humidity !== undefined && latest.humidity !== null ? `${latest.humidity.toFixed(0)}%` : "-"}
          </div>
        </div>

        {/* Wind */}
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.04]">
          <div className="text-[10px] uppercase font-mono text-text-tertiary flex items-center gap-1">
            <Wind className="w-3 h-3" /> Wind
          </div>
          <div className="text-lg font-mono font-bold text-text-primary mt-0.5">
            {latest.wind_speed !== undefined && latest.wind_speed !== null ? `${latest.wind_speed.toFixed(1)} m/s` : "-"}
          </div>
        </div>
      </div>
    </div>
  );
}
