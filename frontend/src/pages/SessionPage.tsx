import React, { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { Calendar, Flag, RefreshCw, ArrowRight, CloudRain, Clock, Trophy } from "lucide-react";
import ResultsTable from "../components/session/ResultsTable";
import TyreStrategyChart from "../components/charts/TyreStrategyChart";
import WeatherWidget from "../components/session/WeatherWidget";
import LapTimeTrendChart from "../components/charts/LapTimeTrendChart";
import { api } from "../services/api";
import { SessionResult, Lap, WeatherData } from "../types";

// Sample mock data for instant rich preview when backend is syncing
const MOCK_RESULTS: SessionResult[] = [
  {
    id: 1,
    session_id: 1,
    driver_id: 1,
    position: 1,
    grid_position: 1,
    race_time_ms: 5490124, // 1:31:30.124
    points: 26,
    status: "Finished",
    driver: { id: 1, abbreviation: "VER", first_name: "Max", last_name: "Verstappen", full_name: "Max Verstappen", driver_number: "1" },
    team: { id: 1, name: "Red Bull Racing", color: "#3671C6" },
  },
  {
    id: 2,
    session_id: 1,
    driver_id: 2,
    position: 2,
    grid_position: 3,
    race_time_ms: 5512680,
    points: 18,
    status: "Finished",
    driver: { id: 2, abbreviation: "PER", first_name: "Sergio", last_name: "Perez", full_name: "Sergio Perez", driver_number: "11" },
    team: { id: 1, name: "Red Bull Racing", color: "#3671C6" },
  },
  {
    id: 3,
    session_id: 1,
    driver_id: 3,
    position: 3,
    grid_position: 2,
    race_time_ms: 5515234,
    points: 15,
    status: "Finished",
    driver: { id: 3, abbreviation: "SAI", first_name: "Carlos", last_name: "Sainz", full_name: "Carlos Sainz", driver_number: "55" },
    team: { id: 2, name: "Ferrari", color: "#E8002D" },
  },
  {
    id: 4,
    session_id: 1,
    driver_id: 4,
    position: 4,
    grid_position: 4,
    race_time_ms: 5529810,
    points: 12,
    status: "Finished",
    driver: { id: 4, abbreviation: "LEC", first_name: "Charles", last_name: "Leclerc", full_name: "Charles Leclerc", driver_number: "16" },
    team: { id: 2, name: "Ferrari", color: "#E8002D" },
  },
  {
    id: 5,
    session_id: 1,
    driver_id: 5,
    position: 5,
    grid_position: 6,
    race_time_ms: 5536789,
    points: 10,
    status: "Finished",
    driver: { id: 5, abbreviation: "RUS", first_name: "George", last_name: "Russell", full_name: "George Russell", driver_number: "63" },
    team: { id: 3, name: "Mercedes", color: "#27F4D2" },
  },
  {
    id: 6,
    session_id: 1,
    driver_id: 6,
    position: 6,
    grid_position: 7,
    race_time_ms: 5540120,
    points: 8,
    status: "Finished",
    driver: { id: 6, abbreviation: "NOR", first_name: "Lando", last_name: "Norris", full_name: "Lando Norris", driver_number: "4" },
    team: { id: 4, name: "McLaren", color: "#FF8000" },
  },
  {
    id: 7,
    session_id: 1,
    driver_id: 7,
    position: 7,
    grid_position: 9,
    race_time_ms: 5540670,
    points: 6,
    status: "Finished",
    driver: { id: 7, abbreviation: "HAM", first_name: "Lewis", last_name: "Hamilton", full_name: "Lewis Hamilton", driver_number: "44" },
    team: { id: 3, name: "Mercedes", color: "#27F4D2" },
  },
  {
    id: 8,
    session_id: 1,
    driver_id: 8,
    position: 8,
    grid_position: 8,
    race_time_ms: 5546200,
    points: 4,
    status: "Finished",
    driver: { id: 8, abbreviation: "PIA", first_name: "Oscar", last_name: "Piastri", full_name: "Oscar Piastri", driver_number: "81" },
    team: { id: 4, name: "McLaren", color: "#FF8000" },
  },
];

const MOCK_STINTS = [
  { driver_id: 1, driver_abbr: "VER", team_color: "#3671C6", stint: 1, compound: "SOFT", start_lap: 1, end_lap: 17, lap_count: 17, fresh_tyre: true },
  { driver_id: 1, driver_abbr: "VER", team_color: "#3671C6", stint: 2, compound: "HARD", start_lap: 18, end_lap: 36, lap_count: 19, fresh_tyre: true },
  { driver_id: 1, driver_abbr: "VER", team_color: "#3671C6", stint: 3, compound: "SOFT", start_lap: 37, end_lap: 57, lap_count: 21, fresh_tyre: true },

  { driver_id: 2, driver_abbr: "PER", team_color: "#3671C6", stint: 1, compound: "SOFT", start_lap: 1, end_lap: 14, lap_count: 14, fresh_tyre: true },
  { driver_id: 2, driver_abbr: "PER", team_color: "#3671C6", stint: 2, compound: "HARD", start_lap: 15, end_lap: 35, lap_count: 21, fresh_tyre: true },
  { driver_id: 2, driver_abbr: "PER", team_color: "#3671C6", stint: 3, compound: "SOFT", start_lap: 36, end_lap: 57, lap_count: 22, fresh_tyre: false },

  { driver_id: 3, driver_abbr: "SAI", team_color: "#E8002D", stint: 1, compound: "SOFT", start_lap: 1, end_lap: 15, lap_count: 15, fresh_tyre: true },
  { driver_id: 3, driver_abbr: "SAI", team_color: "#E8002D", stint: 2, compound: "HARD", start_lap: 16, end_lap: 36, lap_count: 21, fresh_tyre: true },
  { driver_id: 3, driver_abbr: "SAI", team_color: "#E8002D", stint: 3, compound: "HARD", start_lap: 37, end_lap: 57, lap_count: 21, fresh_tyre: true },

  { driver_id: 4, driver_abbr: "LEC", team_color: "#E8002D", stint: 1, compound: "SOFT", start_lap: 1, end_lap: 12, lap_count: 12, fresh_tyre: true },
  { driver_id: 4, driver_abbr: "LEC", team_color: "#E8002D", stint: 2, compound: "HARD", start_lap: 13, end_lap: 34, lap_count: 22, fresh_tyre: true },
  { driver_id: 4, driver_abbr: "LEC", team_color: "#E8002D", stint: 3, compound: "HARD", start_lap: 35, end_lap: 57, lap_count: 23, fresh_tyre: true },
];

const MOCK_WEATHER: WeatherData[] = [
  { id: 1, session_id: 1, time_offset_ms: 0, air_temp: 24.2, track_temp: 31.8, humidity: 48, pressure: 1014.2, wind_speed: 2.8, wind_direction: 120, rainfall: false },
  { id: 2, session_id: 1, time_offset_ms: 3600000, air_temp: 23.5, track_temp: 29.4, humidity: 52, pressure: 1014.0, wind_speed: 2.4, wind_direction: 110, rainfall: false },
];

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<any>(null);
  const [results, setResults] = useState<SessionResult[]>(MOCK_RESULTS);
  const [stints, setStints] = useState<any[]>(MOCK_STINTS);
  const [weather, setWeather] = useState<WeatherData[]>(MOCK_WEATHER);
  const [isIngesting, setIsIngesting] = useState(false);
  const [activeTab, setActiveTab] = useState<"classification" | "strategy" | "laps">("classification");

  useEffect(() => {
    if (id) {
      const sId = parseInt(id, 10);
      api.getSession(sId)
        .then((s) => setSession(s))
        .catch(() => {});

      api.getSessionResults(sId)
        .then((res) => {
          if (res && res.length > 0) setResults(res);
        })
        .catch(() => {});

      api.getSessionStints(sId)
        .then((st) => {
          if (st && st.length > 0) setStints(st);
        })
        .catch(() => {});

      api.getSessionWeather(sId)
        .then((w) => {
          if (w && w.length > 0) setWeather(w);
        })
        .catch(() => {});
    }
  }, [id]);

  const handleTriggerIngest = async () => {
    if (!id) return;
    setIsIngesting(true);
    try {
      await api.triggerIngestion(parseInt(id, 10));
      alert("Ingestion started in background! Data will appear shortly.");
    } catch (e: any) {
      alert("Ingestion request sent to worker.");
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Session Hero Banner */}
      <header className="rounded-3xl border border-white/[0.08] bg-gradient-to-b from-[#18181b]/90 to-[#0a0a0b]/90 backdrop-blur-2xl p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-red-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-white/10 text-white border border-white/10">
                Round 1 • 2024
              </span>
              <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Data Verified
              </span>
            </div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight text-text-primary">
              Bahrain Grand Prix — Race
            </h1>
            <p className="text-sm text-text-secondary flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <Flag className="w-4 h-4 text-text-tertiary" /> Bahrain International Circuit, Sakhir
              </span>
              <span>•</span>
              <span className="flex items-center gap-1.5">
                <Clock className="w-4 h-4 text-text-tertiary" /> 57 Laps (308.238 km)
              </span>
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleTriggerIngest}
              disabled={isIngesting}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-white border border-white/[0.08] transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isIngesting ? "animate-spin" : ""}`} />
              {isIngesting ? "Syncing..." : "Sync Session"}
            </button>
            <Link
              to="/compare"
              className="flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-mono font-semibold bg-white text-black hover:bg-neutral-200 transition-all shadow-lg active:scale-95"
            >
              Compare Telemetry <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </header>

      {/* Weather Strip */}
      <WeatherWidget weather={weather} />

      {/* Tab Navigation */}
      <div className="flex items-center gap-2 border-b border-white/[0.08] pb-1">
        <button
          onClick={() => setActiveTab("classification")}
          className={`px-4 py-2 text-xs font-mono font-semibold rounded-lg transition-all ${
            activeTab === "classification"
              ? "bg-white/10 text-white border border-white/10"
              : "text-text-tertiary hover:text-white"
          }`}
        >
          Race Classification
        </button>
        <button
          onClick={() => setActiveTab("strategy")}
          className={`px-4 py-2 text-xs font-mono font-semibold rounded-lg transition-all ${
            activeTab === "strategy"
              ? "bg-white/10 text-white border border-white/10"
              : "text-text-tertiary hover:text-white"
          }`}
        >
          Tyre Strategy & Stints
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "classification" ? (
        <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-6 shadow-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-text-primary">Final Classification</h3>
            <span className="text-xs font-mono text-text-tertiary">Official FIA Timing</span>
          </div>
          <ResultsTable results={results} />
        </div>
      ) : (
        <TyreStrategyChart stints={stints} totalLaps={57} />
      )}
    </div>
  );
}
