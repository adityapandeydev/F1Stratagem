import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Zap, Trophy, Gauge, Activity, Flag, Calendar, Sparkles } from "lucide-react";
import { api } from "../services/api";
import { Event } from "../types";

export default function HomePage() {
  const [events, setEvents] = useState<Event[]>([]);
  const [activeSeason, setActiveSeason] = useState(2024);

  useEffect(() => {
    api.getEventsBySeason(activeSeason)
      .then((data) => {
        if (data && data.length > 0) setEvents(data);
      })
      .catch(() => {});
  }, [activeSeason]);

  return (
    <div className="space-y-12 pb-16">
      {/* Hero Section */}
      <section className="relative rounded-3xl border border-white/[0.08] bg-gradient-to-br from-[#18181b]/90 via-[#111113]/80 to-[#0a0a0b]/90 backdrop-blur-2xl p-8 md:p-12 shadow-2xl overflow-hidden">
        {/* Ambient Glow Orbs */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-red-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-medium bg-white/[0.06] border border-white/[0.08] text-text-secondary">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Precision Formula 1 Telemetry Platform
          </div>

          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-text-primary leading-tight">
            Stratagem Intelligence. <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-neutral-300 to-neutral-500">
              Beyond the Timing Screen.
            </span>
          </h1>

          <p className="text-base md:text-lg text-text-secondary leading-relaxed max-w-2xl">
            High-frequency telemetry analysis with 3D fluid elevation track maps, delta speed dominance ribbons, throttle trace synchronization, and stint tyre degradation models.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Link
              to="/compare"
              className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold bg-white text-black hover:bg-neutral-200 transition-all shadow-xl active:scale-95"
            >
              Launch 3D Telemetry Studio <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/session/1"
              className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold bg-white/[0.06] hover:bg-white/[0.12] text-white border border-white/[0.08] transition-all"
            >
              Explore Bahrain GP Race
            </Link>
          </div>
        </div>
      </section>

      {/* KPI Stats Grid */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-6 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="text-xs font-mono uppercase tracking-wider text-text-tertiary flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-400" /> Metrics Tracked
          </div>
          <div className="text-3xl font-mono font-extrabold text-text-primary mt-2">168+</div>
          <div className="text-xs text-text-secondary mt-1">Car dynamics, GPS, temperatures & slip</div>
        </div>

        <div className="p-6 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="text-xs font-mono uppercase tracking-wider text-text-tertiary flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" /> Synchronization
          </div>
          <div className="text-3xl font-mono font-extrabold text-text-primary mt-2">10 Hz</div>
          <div className="text-xs text-text-secondary mt-1">Millisecond distance-aligned traces</div>
        </div>

        <div className="p-6 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="text-xs font-mono uppercase tracking-wider text-text-tertiary flex items-center gap-2">
            <Gauge className="w-4 h-4 text-purple-400" /> 3D Elevation
          </div>
          <div className="text-3xl font-mono font-extrabold text-text-primary mt-2">Real Z</div>
          <div className="text-xs text-text-secondary mt-1">Fluid velocity dominance contours</div>
        </div>

        <div className="p-6 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] backdrop-blur-xl">
          <div className="text-xs font-mono uppercase tracking-wider text-text-tertiary flex items-center gap-2">
            <Trophy className="w-4 h-4 text-amber-400" /> Season Coverage
          </div>
          <div className="text-3xl font-mono font-extrabold text-text-primary mt-2">2018 - 2026</div>
          <div className="text-xs text-text-secondary mt-1">Hybrid regulation eras & beyond</div>
        </div>
      </section>

      {/* Featured Race Weekends */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-text-primary">
              2024 Season Calendar
            </h2>
            <p className="text-xs text-text-secondary mt-0.5">
              Select a Grand Prix weekend to explore session classifications and telemetry
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-[#141416] p-1 rounded-xl border border-white/[0.08]">
            {[2024, 2025, 2026].map((yr) => (
              <button
                key={yr}
                onClick={() => setActiveSeason(yr)}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                  activeSeason === yr ? "bg-white/10 text-white" : "text-text-tertiary hover:text-white"
                }`}
              >
                {yr}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <RaceCard
            round={1}
            name="Bahrain Grand Prix"
            circuit="Bahrain International Circuit"
            location="Sakhir, Bahrain"
            date="Mar 2, 2024"
            status="Data Ready"
            winner="M. Verstappen (RBR)"
            link="/session/1"
          />
          <RaceCard
            round={2}
            name="Saudi Arabian Grand Prix"
            circuit="Jeddah Corniche Circuit"
            location="Jeddah, Saudi Arabia"
            date="Mar 9, 2024"
            status="Data Ready"
            winner="M. Verstappen (RBR)"
            link="/session/2"
          />
          <RaceCard
            round={3}
            name="Australian Grand Prix"
            circuit="Albert Park Circuit"
            location="Melbourne, Australia"
            date="Mar 24, 2024"
            status="Data Ready"
            winner="C. Sainz (Ferrari)"
            link="/session/3"
          />
          <RaceCard
            round={4}
            name="Japanese Grand Prix"
            circuit="Suzuka International Racing Course"
            location="Suzuka, Japan"
            date="Apr 7, 2024"
            status="Data Ready"
            winner="M. Verstappen (RBR)"
            link="/session/4"
          />
          <RaceCard
            round={5}
            name="Chinese Grand Prix"
            circuit="Shanghai International Circuit"
            location="Shanghai, China"
            date="Apr 21, 2024"
            status="Data Ready"
            winner="M. Verstappen (RBR)"
            link="/session/5"
          />
          <RaceCard
            round={6}
            name="Miami Grand Prix"
            circuit="Miami International Autodrome"
            location="Miami, USA"
            date="May 5, 2024"
            status="Data Ready"
            winner="L. Norris (McLaren)"
            link="/session/6"
          />
        </div>
      </section>
    </div>
  );
}

interface RaceCardProps {
  round: number;
  name: string;
  circuit: string;
  location: string;
  date: string;
  status: string;
  winner?: string;
  link: string;
}

function RaceCard({ round, name, circuit, location, date, status, winner, link }: RaceCardProps) {
  return (
    <Link
      to={link}
      className="group relative block p-6 rounded-2xl bg-[#0a0a0b]/80 border border-white/[0.08] hover:border-white/[0.2] transition-all backdrop-blur-xl shadow-lg hover:shadow-2xl hover:-translate-y-0.5"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono font-bold text-text-tertiary">
          ROUND {round}
        </span>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-medium">
          {status}
        </span>
      </div>

      <h3 className="text-lg font-bold text-text-primary group-hover:text-white transition-colors">
        {name}
      </h3>
      <p className="text-xs text-text-secondary mt-1">{circuit}</p>

      <div className="mt-4 pt-3 border-t border-white/[0.04] flex items-center justify-between text-xs font-mono">
        <span className="text-text-tertiary">{date}</span>
        {winner && (
          <span className="text-amber-400 font-semibold flex items-center gap-1">
            <Trophy className="w-3 h-3" /> {winner}
          </span>
        )}
      </div>
    </Link>
  );
}
