import React, { useState, useEffect, useMemo } from "react";
import {
  Zap,
  Activity,
  Gauge,
  Flame,
  Sparkles,
  RefreshCw,
  Plus,
  Trash2,
  Calendar,
  Layers,
  CheckCircle2,
  AlertCircle,
  Clock,
  Compass,
} from "lucide-react";
import InteractiveTrack3D from "../components/charts/InteractiveTrack3D";
import TelemetryComparisonChart from "../components/charts/TelemetryComparisonChart";
import { api } from "../services/api";
import { Event, Session, TelemetryPoint, TelemetryDriverMeta } from "../types";

// Master F1 Driver Roster
const ALL_DRIVERS = [
  { abbr: "VER", name: "Max Verstappen", team: "Red Bull Racing", color: "#3671c6" },
  { abbr: "PER", name: "Sergio Perez", team: "Red Bull Racing", color: "#3671c6" },
  { abbr: "LEC", name: "Charles Leclerc", team: "Ferrari", color: "#e8002d" },
  { abbr: "SAI", name: "Carlos Sainz", team: "Ferrari", color: "#e8002d" },
  { abbr: "NOR", name: "Lando Norris", team: "McLaren", color: "#ff8000" },
  { abbr: "PIA", name: "Oscar Piastri", team: "McLaren", color: "#ff8000" },
  { abbr: "RUS", name: "George Russell", team: "Mercedes", color: "#27f4d2" },
  { abbr: "HAM", name: "Lewis Hamilton", team: "Mercedes", color: "#27f4d2" },
  { abbr: "ALO", name: "Fernando Alonso", team: "Aston Martin", color: "#229971" },
  { abbr: "STR", name: "Lance Stroll", team: "Aston Martin", color: "#229971" },
  { abbr: "TSU", name: "Yuki Tsunoda", team: "RB", color: "#6692ff" },
  { abbr: "RIC", name: "Daniel Ricciardo", team: "RB", color: "#6692ff" },
  { abbr: "HUL", name: "Nico Hulkenberg", team: "Haas", color: "#b6babd" },
  { abbr: "MAG", name: "Kevin Magnussen", team: "Haas", color: "#b6babd" },
  { abbr: "ALB", name: "Alexander Albon", team: "Williams", color: "#64c4ff" },
  { abbr: "SAR", name: "Logan Sargeant", team: "Williams", color: "#64c4ff" },
  { abbr: "BOT", name: "Valtteri Bottas", team: "Kick Sauber", color: "#52e252" },
  { abbr: "ZHO", name: "Guanyu Zhou", team: "Kick Sauber", color: "#52e252" },
  { abbr: "GAS", name: "Pierre Gasly", team: "Alpine", color: "#ff87bc" },
  { abbr: "OCO", name: "Esteban Ocon", team: "Alpine", color: "#ff87bc" },
];

export default function ComparePage() {
  // Step 1: Events (Tracks)
  const [events, setEvents] = useState<Event[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<number>(1);

  // Step 2: Sessions
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number>(4);

  // Step 3: Drivers (up to 4 slots)
  const [selectedDrivers, setSelectedDrivers] = useState<string[]>(["VER", "LEC"]);

  // Telemetry state
  const [telemetryData, setTelemetryData] = useState<{
    drivers: TelemetryDriverMeta[];
    telemetry: TelemetryPoint[][];
    time_delta: number[];
  } | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [scrubDistance, setScrubDistance] = useState<number>(0);

  // Load 2024 Events on mount
  useEffect(() => {
    api.getEventsBySeason(2024)
      .then((evts) => {
        if (evts && evts.length > 0) {
          setEvents(evts);
          // Default to Round 1 (Bahrain)
          const bhr = evts.find((e) => e.round_number === 1) || evts[0];
          setSelectedEventId(bhr.id);
        }
      })
      .catch((err) => {
        console.error("Failed to load events:", err);
      });
  }, []);

  // Load Sessions when Event changes
  useEffect(() => {
    if (!selectedEventId) return;
    api.getSessionsByEvent(selectedEventId)
      .then((sessList) => {
        setSessions(sessList);
        // Default to Qualifying session if available, else first
        const qual = sessList.find(
          (s) => s.session_type === "qualifying" || s.session_name.toLowerCase().includes("qual")
        );
        if (qual) {
          setSelectedSessionId(qual.id);
        } else if (sessList.length > 0) {
          setSelectedSessionId(sessList[0].id);
        }
      })
      .catch((err) => {
        console.error("Failed to load sessions:", err);
      });
  }, [selectedEventId]);

  // Fetch Multi-Driver Telemetry when Session or Drivers change
  useEffect(() => {
    if (!selectedSessionId || selectedDrivers.length < 2) return;

    let cancelled = false;
    setIsLoading(true);
    setErrorMessage(null);

    const driverReqs = selectedDrivers.map((abbr) => ({ driver: abbr, lap: 0 }));

    api.getMultiTelemetryComparison(selectedSessionId, driverReqs)
      .then((res: any) => {
        if (cancelled) return;
        const data = res?.data || res;
        if (data && data.telemetry && data.telemetry.length >= 2) {
          setTelemetryData(data);
          setIsLoading(false);
        } else {
          setErrorMessage("No telemetry data returned for this session combination.");
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("Telemetry fetch error:", err);
          setErrorMessage(
            err.response?.data?.error || "Failed to load telemetry from backend. Ensure worker is running."
          );
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSessionId, selectedDrivers]);

  // Handle Driver Slot Selection
  const handleDriverChange = (slotIndex: number, newAbbr: string) => {
    const next = [...selectedDrivers];
    next[slotIndex] = newAbbr;
    setSelectedDrivers(next);
  };

  // Add Driver Slot (up to 4)
  const handleAddDriver = () => {
    if (selectedDrivers.length >= 4) return;
    // Pick first driver not currently selected
    const available = ALL_DRIVERS.find((d) => !selectedDrivers.includes(d.abbr));
    if (available) {
      setSelectedDrivers([...selectedDrivers, available.abbr]);
    }
  };

  // Remove Driver Slot (min 2)
  const handleRemoveDriver = (slotIndex: number) => {
    if (selectedDrivers.length <= 2) return;
    const next = selectedDrivers.filter((_, idx) => idx !== slotIndex);
    setSelectedDrivers(next);
  };

  // Selected Event Object
  const currentEvent = useMemo(
    () => events.find((e) => e.id === selectedEventId),
    [events, selectedEventId]
  );

  // Selected Session Object
  const currentSession = useMemo(
    () => sessions.find((s) => s.id === selectedSessionId),
    [sessions, selectedSessionId]
  );

  // Format milliseconds into mm:ss.sss
  const formatTime = (ms?: number) => {
    if (!ms) return "--:--.---";
    const mins = Math.floor(ms / 60000);
    const secs = ((ms % 60000) / 1000).toFixed(3);
    return `${mins}:${secs.padStart(6, "0")}`;
  };

  return (
    <div className="space-y-7 animate-fade-in pb-12">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.06] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-md shadow-red-500/50" />
            <span className="text-[11px] font-mono font-bold tracking-widest text-neutral-400 uppercase">
              Phase 1 Telemetry Engine &bull; FastF1 Studio
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-black tracking-tight text-white flex items-center gap-3">
            Multi-Driver Telemetry & 3D Circuit Studio
          </h1>
          <p className="text-xs md:text-sm text-neutral-400 mt-1 max-w-2xl">
            Compare authentic GPS racing lines, throttle response, braking points, and speed dominance across up to 4 drivers simultaneously.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>FastF1 GPS Sync Active</span>
          </div>
        </div>
      </div>

      {/* 3-Step Cascade Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 p-5 rounded-2xl border border-white/[0.08] bg-[#0d0d11]/90 backdrop-blur-xl shadow-2xl">
        {/* Step 1: Track / Event */}
        <div className="lg:col-span-4 space-y-1.5">
          <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
              1
            </span>
            Select Grand Prix Track
          </label>
          <select
            value={selectedEventId}
            onChange={(e) => setSelectedEventId(Number(e.target.value))}
            className="w-full bg-[#16161a] border border-white/[0.1] hover:border-white/[0.2] focus:border-red-500 rounded-xl px-3.5 py-2.5 text-xs md:text-sm font-medium text-white transition-all outline-none cursor-pointer"
          >
            {events.map((ev) => (
              <option key={ev.id} value={ev.id}>
                R{ev.round_number} &bull; {ev.event_name} ({ev.country || "GP"})
              </option>
            ))}
          </select>
          {currentEvent && (
            <p className="text-[10px] font-mono text-neutral-500 px-1">
              Location: {currentEvent.location || "Sakhir"}, {currentEvent.country}
            </p>
          )}
        </div>

        {/* Step 2: Session */}
        <div className="lg:col-span-3 space-y-1.5">
          <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
              2
            </span>
            Select Session
          </label>
          <select
            value={selectedSessionId}
            onChange={(e) => setSelectedSessionId(Number(e.target.value))}
            className="w-full bg-[#16161a] border border-white/[0.1] hover:border-white/[0.2] focus:border-red-500 rounded-xl px-3.5 py-2.5 text-xs md:text-sm font-medium text-white transition-all outline-none cursor-pointer"
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.session_name}
              </option>
            ))}
          </select>
          {currentSession && (
            <p className="text-[10px] font-mono text-neutral-500 px-1">
              Format: {currentSession.session_type?.toUpperCase()}
            </p>
          )}
        </div>

        {/* Step 3: Driver Selection Slots */}
        <div className="lg:col-span-5 space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
                3
              </span>
              Drivers to Compare ({selectedDrivers.length}/4)
            </label>
            {selectedDrivers.length < 4 && (
              <button
                onClick={handleAddDriver}
                className="flex items-center gap-1 text-[11px] font-medium text-red-400 hover:text-red-300 transition-colors"
              >
                <Plus className="w-3 h-3" />
                Add Driver
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {selectedDrivers.map((currAbbr, slotIdx) => {
              const driverObj = ALL_DRIVERS.find((d) => d.abbr === currAbbr);

              // Filter out drivers already chosen in other slots (Mutual Exclusion)
              const availableDrivers = ALL_DRIVERS.filter(
                (d) => d.abbr === currAbbr || !selectedDrivers.includes(d.abbr)
              );

              return (
                <div
                  key={slotIdx}
                  className="flex items-center gap-2 p-1.5 bg-[#16161a] rounded-xl border border-white/[0.08]"
                >
                  <span
                    className="w-3 h-3 rounded-full shrink-0 ml-1.5"
                    style={{ backgroundColor: driverObj?.color || "#ffffff" }}
                  />
                  <select
                    value={currAbbr}
                    onChange={(e) => handleDriverChange(slotIdx, e.target.value)}
                    className="flex-1 bg-transparent text-xs font-mono font-bold text-white outline-none cursor-pointer"
                  >
                    {availableDrivers.map((d) => (
                      <option key={d.abbr} value={d.abbr} className="bg-[#16161a] text-white">
                        {d.abbr} &bull; {d.name} ({d.team})
                      </option>
                    ))}
                  </select>

                  {selectedDrivers.length > 2 && (
                    <button
                      onClick={() => handleRemoveDriver(slotIdx)}
                      className="p-1 text-neutral-500 hover:text-red-400 rounded-lg transition-colors shrink-0"
                      title="Remove Driver"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Driver Metric Overview Cards */}
      {telemetryData && telemetryData.drivers && telemetryData.drivers.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3">
          {telemetryData.drivers.map((drv, idx) => {
            const isReference = idx === 0;
            const refLapMs = telemetryData.drivers[0]?.lap_time_ms || 0;
            const deltaMs = drv.lap_time_ms - refLapMs;

            return (
              <div
                key={drv.abbreviation}
                className="p-4 rounded-2xl border bg-[#0d0d11]/80 backdrop-blur-md shadow-lg transition-all"
                style={{ borderColor: `${drv.color}40` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: drv.color }} />
                    <span className="text-base font-extrabold tracking-tight text-white font-mono">
                      {drv.abbreviation}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-white/[0.05] text-neutral-400">
                    Lap {drv.lap_number}
                  </span>
                </div>

                <div className="text-xl font-mono font-black text-white">
                  {formatTime(drv.lap_time_ms)}
                </div>

                <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-white/[0.06]">
                  <span className="text-neutral-400 text-[11px] truncate max-w-[100px]">
                    {drv.team_name}
                  </span>
                  <span className="font-mono font-semibold" style={{ color: drv.color }}>
                    {isReference ? "Reference" : `+${(deltaMs / 1000).toFixed(3)}s`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="p-16 rounded-2xl border border-white/[0.08] bg-[#0a0a0c]/80 backdrop-blur-xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="relative w-12 h-12">
            <div className="absolute inset-0 rounded-full border-2 border-red-500/20 animate-ping" />
            <div className="w-12 h-12 rounded-full border-2 border-red-500 border-t-transparent animate-spin" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">
              Extracting FastF1 Telemetry & GPS Coordinates...
            </h3>
            <p className="text-xs text-neutral-400 mt-1 font-mono">
              Loading authentic {selectedDrivers.join(", ")} telemetry streams from Bahrain GP
            </p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-center gap-3 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* 3D Track & Telemetry Visualizations */}
      {!isLoading && telemetryData && telemetryData.drivers && telemetryData.drivers.length >= 2 && (
        <>
          {/* 3D Interactive Track HUD */}
          <InteractiveTrack3D
            drivers={telemetryData.drivers}
            telemetryStreams={telemetryData.telemetry}
            currentDistance={scrubDistance}
            onDistanceChange={setScrubDistance}
          />

          {/* Synchronized ECharts Telemetry Traces */}
          <TelemetryComparisonChart
            drivers={telemetryData.drivers}
            telemetryStreams={telemetryData.telemetry}
            timeDelta={telemetryData.time_delta}
            currentDistance={scrubDistance}
            onDistanceHover={setScrubDistance}
          />
        </>
      )}
    </div>
  );
}
