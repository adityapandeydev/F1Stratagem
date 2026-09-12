import React, { useState, useEffect, useMemo } from "react";
import {
  Plus,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Calendar,
  Layers,
  ChevronDown,
} from "lucide-react";
import InteractiveTrack3D from "../components/charts/InteractiveTrack3D";
import TelemetryComparisonChart from "../components/charts/TelemetryComparisonChart";
import CustomSelect from "../components/ui/CustomSelect";
import { api } from "../services/api";
import { Event, Session, TelemetryPoint, TelemetryDriverMeta } from "../types";

// Master F1 Driver Roster with official liveries (Kick Sauber 2024 neon green, Aston Martin racing green)
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

const AVAILABLE_YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018];

interface DriverSelectionSlot {
  driver: string;
  lap: number; // 0 = Fastest, >0 = specific lap
}

export default function ComparePage() {
  // Step 0: Year
  const [selectedYear, setSelectedYear] = useState<number>(2024);

  // Step 1: Track / Event
  const [events, setEvents] = useState<Event[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<number>(1);

  // Step 2: Session
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number>(4);

  // Step 3: Drivers with Lap Selection (up to 4)
  const [selectedDrivers, setSelectedDrivers] = useState<DriverSelectionSlot[]>([
    { driver: "VER", lap: 0 },
    { driver: "PER", lap: 0 },
    { driver: "LEC", lap: 0 },
    { driver: "SAI", lap: 0 },
  ]);

  // Telemetry state
  const [telemetryData, setTelemetryData] = useState<{
    drivers: TelemetryDriverMeta[];
    telemetry: TelemetryPoint[][];
    time_delta: number[];
  } | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [scrubDistance, setScrubDistance] = useState<number>(0);

  // Load Events when Year changes
  useEffect(() => {
    api.getEventsBySeason(selectedYear)
      .then((evts) => {
        if (evts && evts.length > 0) {
          setEvents(evts);
          const bhr = evts.find((e) => e.round_number === 1) || evts[0];
          setSelectedEventId(bhr.id);
        } else {
          setEvents([]);
        }
      })
      .catch((err) => {
        console.error("Failed to load events for year:", selectedYear, err);
      });
  }, [selectedYear]);

  // Load Sessions when Event changes
  useEffect(() => {
    if (!selectedEventId) return;
    api.getSessionsByEvent(selectedEventId)
      .then((sessList) => {
        setSessions(sessList);
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

  // Fetch Multi-Driver Telemetry
  useEffect(() => {
    if (!selectedSessionId || selectedDrivers.length < 2) return;

    let cancelled = false;
    setIsLoading(true);
    setErrorMessage(null);

    const driverReqs = selectedDrivers.map((slot) => ({ driver: slot.driver, lap: slot.lap }));

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

  // Teammate Color Disambiguation Algorithm
  // If teammates are selected, 1st retains team livery; 2nd gets Pure White (#ffffff); 2nd pair gets Electric Gold (#ffd700).
  const resolvedDrivers: TelemetryDriverMeta[] = useMemo(() => {
    if (!telemetryData || !telemetryData.drivers) return [];
    const teamCounts: Record<string, number> = {};
    const neutralAccents = ["#ffffff", "#ffd700", "#e2e8f0", "#38bdf8"];
    let accentIdx = 0;

    return telemetryData.drivers.map((drv) => {
      const team = drv.team_name || "Unknown";
      teamCounts[team] = (teamCounts[team] || 0) + 1;

      let color = drv.color;
      if (teamCounts[team] > 1) {
        color = neutralAccents[accentIdx % neutralAccents.length];
        accentIdx++;
      }

      return {
        ...drv,
        color,
      };
    });
  }, [telemetryData]);

  // Driver Slot Selection Handler
  const handleDriverChange = (slotIndex: number, newAbbr: string) => {
    const next = [...selectedDrivers];
    next[slotIndex] = { ...next[slotIndex], driver: newAbbr };
    setSelectedDrivers(next);
  };

  const handleLapChange = (slotIndex: number, newLap: number) => {
    const next = [...selectedDrivers];
    next[slotIndex] = { ...next[slotIndex], lap: newLap };
    setSelectedDrivers(next);
  };

  const handleAddDriver = () => {
    if (selectedDrivers.length >= 4) return;
    const usedAbbrs = selectedDrivers.map((d) => d.driver);
    const available = ALL_DRIVERS.find((d) => !usedAbbrs.includes(d.abbr));
    if (available) {
      setSelectedDrivers([...selectedDrivers, { driver: available.abbr, lap: 0 }]);
    }
  };

  const handleRemoveDriver = (slotIndex: number) => {
    if (selectedDrivers.length <= 2) return;
    const next = selectedDrivers.filter((_, idx) => idx !== slotIndex);
    setSelectedDrivers(next);
  };

  const currentEvent = useMemo(
    () => events.find((e) => e.id === selectedEventId),
    [events, selectedEventId]
  );

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === selectedSessionId),
    [sessions, selectedSessionId]
  );

  const formatTime = (ms?: number) => {
    if (!ms) return "--:--.---";
    const mins = Math.floor(ms / 60000);
    const secs = ((ms % 60000) / 1000).toFixed(3);
    return `${mins}:${secs.padStart(6, "0")}`;
  };

  return (
    <div className="space-y-6 animate-fade-in pb-12">
      {/* Minimalist Apple-Grade Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.07] pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-red-500 shadow-sm shadow-red-500/50 animate-pulse" />
            <span className="text-[10px] font-mono font-bold tracking-widest text-neutral-400 uppercase">
              Telemetry Studio
            </span>
          </div>
          <h1 className="text-2xl font-black tracking-tight text-white">
            Telemetry Analysis
          </h1>
          <p className="text-xs text-neutral-400 mt-0.5">
            Compare authentic GPS racing lines, throttle response, braking points, and speed dominance.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>FastF1 GPS Active</span>
          </div>
        </div>
      </div>

      {/* 2-Line Cascade Selector: Line 1 (Season -> Track -> Session), Line 2 (Driver Selection) */}
      <div className="p-4.5 rounded-2xl border border-white/[0.08] bg-[#0d0d12]/90 backdrop-blur-2xl shadow-xl space-y-4 relative z-30">
        {/* Line 1: Session Cascade Selector */}
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-3.5 pb-3.5 border-b border-white/[0.06]">
          {/* Step 1: Year */}
          <div className="sm:col-span-3 lg:col-span-2 space-y-1.5">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
                1
              </span>
              Season Year
            </label>
            <CustomSelect
              options={AVAILABLE_YEARS.map((yr) => ({
                value: yr,
                label: `${yr} Season`,
              }))}
              value={selectedYear}
              onChange={(val) => setSelectedYear(Number(val))}
              className="w-full"
            />
          </div>

          {/* Step 2: Track / Event */}
          <div className="sm:col-span-5 lg:col-span-6 space-y-1.5">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
                2
              </span>
              Grand Prix Track
            </label>
            <CustomSelect
              options={events.map((ev) => ({
                value: ev.id,
                label: `R${ev.round_number} • ${ev.event_name}`,
                subLabel: ev.country || "GP",
              }))}
              value={selectedEventId}
              onChange={(val) => setSelectedEventId(Number(val))}
              placeholder="Select Grand Prix Track..."
              className="w-full"
            />
          </div>

          {/* Step 3: Session */}
          <div className="sm:col-span-4 lg:col-span-4 space-y-1.5">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
                3
              </span>
              Session
            </label>
            <CustomSelect
              options={sessions.map((s) => ({
                value: s.id,
                label: s.session_name,
              }))}
              value={selectedSessionId}
              onChange={(val) => setSelectedSessionId(Number(val))}
              placeholder="Select Session..."
              className="w-full"
            />
          </div>
        </div>

        {/* Line 2: Driver Selection Slots (Full Width Dedicated Row) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-white/10 text-white flex items-center justify-center text-[10px]">
                4
              </span>
              Driver Telemetry Slots ({selectedDrivers.length}/4)
            </label>
            {selectedDrivers.length < 4 && (
              <button
                onClick={handleAddDriver}
                className="flex items-center gap-1 text-[11px] font-semibold text-red-400 hover:text-white px-2.5 py-1 rounded-lg hover:bg-red-500/20 transition-all border border-red-500/20 cursor-pointer"
              >
                <Plus className="w-3 h-3" />
                Add Driver Slot
              </button>
            )}
          </div>

          {/* Spacious 2-Column Driver Slots Layout (2 drivers per line, so every driver name and lap time is 100% visible) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {selectedDrivers.map((slot, slotIdx) => {
              const driverObj = ALL_DRIVERS.find((d) => d.abbr === slot.driver);
              const color = resolvedDrivers[slotIdx]?.color || driverObj?.color || "#ffffff";
              const driverMeta = resolvedDrivers.find((d) => d.abbreviation === slot.driver);
              const availableLaps = driverMeta?.available_laps;

              const usedDrivers = selectedDrivers.map((s) => s.driver);
              const availableDrivers = ALL_DRIVERS.filter(
                (d) => d.abbr === slot.driver || !usedDrivers.includes(d.abbr)
              );

              return (
                <div
                  key={slotIdx}
                  style={{ zIndex: selectedDrivers.length - slotIdx + 10 }}
                  className="relative flex items-center justify-between gap-3 px-3.5 py-2.5 bg-[#14141c] rounded-2xl border border-white/[0.08] hover:border-white/[0.16] transition-all shadow-sm"
                >
                  {/* Demarcated Driver Selector Pill */}
                  <div className="flex-1 min-w-0">
                    <CustomSelect
                      pillMode={true}
                      options={availableDrivers.map((d) => ({
                        value: d.abbr,
                        label: `${d.abbr} • ${d.name}`,
                        color: d.color,
                      }))}
                      value={slot.driver}
                      onChange={(val) => handleDriverChange(slotIdx, String(val))}
                      className="w-full"
                    />
                  </div>

                  {/* Demarcated Lap Selector Pill - Aligned directly below the box */}
                  <div className="shrink-0 w-[230px] sm:w-[245px]">
                    <CustomSelect
                      pillMode={true}
                      options={[
                        {
                          value: 0,
                          label: `Best Lap ${driverMeta?.lap_time_ms ? `(${formatTime(driverMeta.lap_time_ms)})` : ""}`,
                          isFastest: true,
                        },
                        ...(availableLaps && availableLaps.length > 0
                          ? availableLaps.map((l) => ({
                              value: l.lap,
                              label: l.label,
                              isFastest: l.is_pb,
                            }))
                          : [2, 5, 8, 13, 16].map((lNum) => ({
                              value: lNum,
                              label: `Lap ${lNum}`,
                            }))),
                      ]}
                      value={slot.lap}
                      onChange={(val) => handleLapChange(slotIdx, Number(val))}
                      className="w-full"
                    />
                  </div>

                  {/* Trash Icon Button */}
                  {selectedDrivers.length > 2 && (
                    <button
                      onClick={() => handleRemoveDriver(slotIdx)}
                      className="p-1.5 text-neutral-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors shrink-0 cursor-pointer"
                      title="Remove Driver Slot"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Driver Metric Overview Cards with Teammate Disambiguation */}
      {resolvedDrivers.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3 relative z-10">
          {resolvedDrivers.map((drv, idx) => {
            const isReference = idx === 0;
            const refLapMs = resolvedDrivers[0]?.lap_time_ms || 0;
            const deltaMs = drv.lap_time_ms - refLapMs;
            const isFastest = selectedDrivers[idx]?.lap === 0;

            return (
              <div
                key={drv.abbreviation}
                className="p-3.5 rounded-2xl border bg-[#0d0d12]/85 backdrop-blur-md shadow-md transition-all"
                style={{ borderColor: `${drv.color}40` }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: drv.color }} />
                    <span className="text-sm font-extrabold tracking-tight text-white font-mono">
                      {drv.abbreviation}
                    </span>
                  </div>
                  <span
                    className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-white/[0.05] text-neutral-300 border border-white/[0.06] flex items-center gap-1"
                    title={isFastest ? "Driver's fastest lap in session" : `Lap ${drv.lap_number}`}
                  >
                    {isFastest ? `⚡ Best (L${drv.lap_number})` : `Lap ${drv.lap_number}`}
                  </span>
                </div>

                <div className="text-lg font-mono font-black text-white">
                  {formatTime(drv.lap_time_ms)}
                </div>

                <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-white/[0.06]">
                  <span className="text-neutral-400 text-[11px] truncate max-w-[110px]">
                    {drv.team_name}
                  </span>
                  <span className="font-mono font-semibold text-xs" style={{ color: drv.color }}>
                    {isReference ? "Reference" : `${deltaMs >= 0 ? "+" : ""}${(deltaMs / 1000).toFixed(3)}s`}
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
              Loading FastF1 Telemetry & GPS Coordinates...
            </h3>
            <p className="text-xs text-neutral-400 mt-1 font-mono">
              Fetching {selectedDrivers.map((d) => d.driver).join(", ")} telemetry streams
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
      {!isLoading && resolvedDrivers.length >= 2 && telemetryData?.telemetry && (
        <>
          {/* 3D Interactive Track HUD (TrackSims Grade) */}
          <InteractiveTrack3D
            drivers={resolvedDrivers}
            telemetryStreams={telemetryData.telemetry}
            timeDelta={telemetryData.time_delta}
            currentDistance={scrubDistance}
            onDistanceChange={setScrubDistance}
          />

          {/* Synchronized ECharts Telemetry Traces (3 Subplots) */}
          <TelemetryComparisonChart
            drivers={resolvedDrivers}
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
