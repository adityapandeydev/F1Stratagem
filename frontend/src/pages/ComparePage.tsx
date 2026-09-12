import React, { useState, useEffect } from "react";
import { Zap, Activity, Gauge, Flame, Sparkles } from "lucide-react";
import InteractiveTrack3D from "../components/charts/InteractiveTrack3D";
import TelemetryComparisonChart from "../components/charts/TelemetryComparisonChart";
import { api } from "../services/api";
import { TelemetryPoint } from "../types";

// Fallback high-fidelity sample telemetry for instant wow experience
function generateSampleTelemetry(driver1Color: string, driver2Color: string) {
  const points1: TelemetryPoint[] = [];
  const points2: TelemetryPoint[] = [];
  const delta: number[] = [];

  const totalPoints = 350;
  const trackLength = 5412; // Bahrain circuit length (m)

  // Track layout simulator (parametric Bahrain GP shape with elevation)
  for (let i = 0; i < totalPoints; i++) {
    const d = (i / totalPoints) * trackLength;
    const t = (i / totalPoints) * Math.PI * 2;

    // Bahrain track shape approximation with chicane & hairpins
    const x = Math.sin(t) * 1200 + Math.sin(t * 3) * 350;
    const y = Math.cos(t) * 1000 + Math.cos(t * 2) * 450;
    // Real elevation change (Z)
    const z = Math.sin(t * 2.5) * 45 + Math.cos(t * 5) * 20;

    // Corner simulation
    const isCorner = Math.sin(t * 4) < -0.3 || Math.cos(t * 3) > 0.6;
    let baseSpd = isCorner ? 85 + Math.random() * 30 : 285 + Math.random() * 45;
    
    // Driver 1 (e.g. VER) higher apex speed
    const spd1 = Math.round(baseSpd + (isCorner ? 4.5 : -2.0) + Math.sin(i * 0.1) * 3);
    // Driver 2 (e.g. LEC) higher top speed on straights
    const spd2 = Math.round(baseSpd + (isCorner ? -3.0 : 5.2) + Math.cos(i * 0.1) * 3);

    const thr1 = isCorner ? 25 : 100;
    const thr2 = isCorner ? 15 : 100;
    const brk1 = isCorner ? 1 : 0;
    const brk2 = isCorner ? 1 : 0;

    const gear1 = Math.min(8, Math.max(2, Math.floor(spd1 / 42)));
    const gear2 = Math.min(8, Math.max(2, Math.floor(spd2 / 42)));

    const rpm1 = Math.round(10500 + (spd1 % 40) * 80);
    const rpm2 = Math.round(10600 + (spd2 % 40) * 80);

    const drs1 = !isCorner && d > 1200 && d < 2200 ? 12 : 0;
    const drs2 = !isCorner && d > 1200 && d < 2200 ? 12 : 0;

    points1.push({ d: Math.round(d), spd: spd1, thr: thr1, brk: brk1, gear: gear1, rpm: rpm1, drs: drs1, x, y, z });
    points2.push({ d: Math.round(d), spd: spd2, thr: thr2, brk: brk2, gear: gear2, rpm: rpm2, drs: drs2, x, y, z });

    // Delta time accumulating
    const deltaDiff = ((spd2 - spd1) / 3600) * 0.08;
    const prevDelta = delta.length > 0 ? delta[delta.length - 1] : 0;
    delta.push(Number((prevDelta + deltaDiff).toFixed(3)));
  }

  return { points1, points2, delta };
}

export default function ComparePage() {
  const [driver1, setDriver1] = useState("VER");
  const [driver2, setDriver2] = useState("LEC");
  const [selectedLap1, setSelectedLap1] = useState(0); // 0 = fastest
  const [selectedLap2, setSelectedLap2] = useState(0);
  const [scrubDistance, setScrubDistance] = useState(0);
  const [activeTab, setActiveTab] = useState<"telemetry" | "sectors" | "corners">("telemetry");

  const DRIVER_COLORS: Record<string, { color: string; team: string; name: string }> = {
    VER: { color: "#3671C6", team: "Red Bull Racing", name: "Max Verstappen" },
    LEC: { color: "#E8002D", team: "Ferrari", name: "Charles Leclerc" },
    HAM: { color: "#27F4D2", team: "Mercedes", name: "Lewis Hamilton" },
    NOR: { color: "#FF8000", team: "McLaren", name: "Lando Norris" },
    PIA: { color: "#FF8000", team: "McLaren", name: "Oscar Piastri" },
    RUS: { color: "#27F4D2", team: "Mercedes", name: "George Russell" },
    SAI: { color: "#E8002D", team: "Ferrari", name: "Carlos Sainz" },
    ALO: { color: "#229971", team: "Aston Martin", name: "Fernando Alonso" },
  };

  const d1Meta = {
    abbreviation: driver1,
    name: DRIVER_COLORS[driver1]?.name || driver1,
    color: DRIVER_COLORS[driver1]?.color || "#3b82f6",
    lapNumber: selectedLap1 || 18,
    lapTimeMs: 90240, // 1:30.240
  };

  const d2Meta = {
    abbreviation: driver2,
    name: DRIVER_COLORS[driver2]?.name || driver2,
    color: DRIVER_COLORS[driver2]?.color || "#ef4444",
    lapNumber: selectedLap2 || 19,
    lapTimeMs: 90468, // 1:30.468
  };

  const sample = React.useMemo(
    () => generateSampleTelemetry(d1Meta.color, d2Meta.color),
    [driver1, driver2]
  );

  return (
    <div className="space-y-8 pb-12">
      {/* Header with Title and Driver Switcher Pills */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.06] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-white/[0.08] text-white tracking-wider uppercase">
              Pro Telemetry Studio
            </span>
            <span className="text-xs text-text-tertiary">Bahrain Grand Prix — Qualifying Q3</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-text-primary">
            Telemetry & 3D Track Analysis
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            Compare throttle, braking thresholds, apex velocities, and track elevation deltas in real-time.
          </p>
        </div>

        {/* Driver Selection Controls */}
        <div className="flex items-center gap-3 bg-[#111113] p-1.5 rounded-2xl border border-white/[0.08] shadow-lg">
          {/* Driver 1 Select */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.04]">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d1Meta.color }} />
            <select
              value={driver1}
              onChange={(e) => setDriver1(e.target.value)}
              className="bg-transparent text-xs font-mono font-bold text-white focus:outline-none cursor-pointer"
            >
              {Object.keys(DRIVER_COLORS).map((abbr) => (
                <option key={abbr} value={abbr} className="bg-[#18181b] text-white">
                  {abbr} — {DRIVER_COLORS[abbr].name}
                </option>
              ))}
            </select>
          </div>

          <span className="text-text-tertiary font-mono font-bold text-xs">VS</span>

          {/* Driver 2 Select */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.04]">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d2Meta.color }} />
            <select
              value={driver2}
              onChange={(e) => setDriver2(e.target.value)}
              className="bg-transparent text-xs font-mono font-bold text-white focus:outline-none cursor-pointer"
            >
              {Object.keys(DRIVER_COLORS).map((abbr) => (
                <option key={abbr} value={abbr} className="bg-[#18181b] text-white">
                  {abbr} — {DRIVER_COLORS[abbr].name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* 3D Fluid Track Speed Visualizer */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h2 className="text-lg font-bold text-text-primary">
              Interactive 3D Track Elevation & Velocity Dominance
            </h2>
          </div>
          <span className="text-xs text-text-tertiary font-mono">
            Drag to rotate • Scroll to zoom • Interactive Scrubbing
          </span>
        </div>

        <InteractiveTrack3D
          telemetry1={sample.points1}
          telemetry2={sample.points2}
          driver1={d1Meta}
          driver2={d2Meta}
          currentDistance={scrubDistance}
          onDistanceChange={(d) => setScrubDistance(d)}
        />
      </section>

      {/* Synchronized Multi-Metric Telemetry Charts */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <h2 className="text-lg font-bold text-text-primary">
              Synchronized Telemetry Traces
            </h2>
          </div>
          <span className="text-xs font-mono text-text-tertiary">
            Hover over curves to sync with 3D track car position
          </span>
        </div>

        <TelemetryComparisonChart
          telemetry1={sample.points1}
          telemetry2={sample.points2}
          driver1={d1Meta}
          driver2={d2Meta}
          timeDelta={sample.delta}
          currentDistance={scrubDistance}
          onDistanceHover={(d) => setScrubDistance(d)}
        />
      </section>

      {/* Deep Dive Performance Metric Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
            <Gauge className="w-4 h-4 text-purple-400" /> Top Speed Trap
          </div>
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-2xl font-mono font-bold text-text-primary">328.4 <span className="text-xs font-normal text-text-tertiary">km/h</span></div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d1Meta.color }}>{d1Meta.abbreviation}</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-text-primary">331.2 <span className="text-xs font-normal text-text-tertiary">km/h</span></div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d2Meta.color }}>{d2Meta.abbreviation} (+2.8)</div>
            </div>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
            <Flame className="w-4 h-4 text-amber-400" /> Minimum Corner Speed
          </div>
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-2xl font-mono font-bold text-text-primary">78.6 <span className="text-xs font-normal text-text-tertiary">km/h</span></div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d1Meta.color }}>{d1Meta.abbreviation} (+3.2)</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-text-primary">75.4 <span className="text-xs font-normal text-text-tertiary">km/h</span></div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d2Meta.color }}>{d2Meta.abbreviation}</div>
            </div>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
            <Zap className="w-4 h-4 text-emerald-400" /> Full Throttle %
          </div>
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-2xl font-mono font-bold text-text-primary">68.4%</div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d1Meta.color }}>{d1Meta.abbreviation}</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-text-primary">69.1%</div>
              <div className="text-xs font-mono mt-0.5" style={{ color: d2Meta.color }}>{d2Meta.abbreviation} (+0.7%)</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
