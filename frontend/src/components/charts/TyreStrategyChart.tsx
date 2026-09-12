import React from "react";

export interface StintData {
  driver_id: number;
  driver_name?: string;
  driver_abbr?: string;
  team_color?: string;
  stint: number;
  compound: string;
  start_lap: number;
  end_lap: number;
  lap_count: number;
  fresh_tyre?: boolean;
}

interface TyreStrategyChartProps {
  stints: StintData[];
  totalLaps?: number;
}

const COMPOUND_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  SOFT: { bg: "#ef4444", text: "#ffffff", label: "S" },
  MEDIUM: { bg: "#eab308", text: "#000000", label: "M" },
  HARD: { bg: "#ffffff", text: "#000000", label: "H" },
  INTERMEDIATE: { bg: "#22c55e", text: "#ffffff", label: "I" },
  WET: { bg: "#3b82f6", text: "#ffffff", label: "W" },
};

export default function TyreStrategyChart({ stints, totalLaps = 57 }: TyreStrategyChartProps) {
  // Group stints by driver
  const driverMap = new Map<number, StintData[]>();
  stints.forEach((s) => {
    if (!driverMap.has(s.driver_id)) {
      driverMap.set(s.driver_id, []);
    }
    driverMap.get(s.driver_id)!.push(s);
  });

  const drivers = Array.from(driverMap.entries());

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-base font-semibold text-text-primary">Tyre Strategy & Stints</h3>
          <p className="text-xs text-text-secondary mt-0.5">
            Compound progression, stint lengths, and pit windows across {totalLaps} laps
          </p>
        </div>

        {/* Compound Legend */}
        <div className="flex items-center gap-3">
          {Object.entries(COMPOUND_COLORS).map(([name, conf]) => (
            <div key={name} className="flex items-center gap-1.5 text-xs font-mono">
              <span
                className="w-4 h-4 rounded-full flex items-center justify-center font-bold text-[10px]"
                style={{ backgroundColor: conf.bg, color: conf.text }}
              >
                {conf.label}
              </span>
              <span className="text-text-tertiary capitalize text-[11px]">{name.toLowerCase()}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Driver Stint Bars */}
      <div className="space-y-3">
        {drivers.map(([driverId, driverStints]) => {
          const firstStint = driverStints[0];
          const abbr = firstStint.driver_abbr || `D${driverId}`;
          const teamColor = firstStint.team_color || "#3b82f6";

          return (
            <div key={driverId} className="flex items-center gap-4">
              {/* Driver Abbr & Team Bar */}
              <div className="flex items-center gap-2 w-20 shrink-0">
                <span className="w-1.5 h-6 rounded-full" style={{ backgroundColor: teamColor }} />
                <span className="font-mono text-xs font-semibold text-text-primary">{abbr}</span>
              </div>

              {/* Stint Timeline Track */}
              <div className="flex-1 h-7 bg-white/[0.03] rounded-lg p-1 flex items-center gap-1 border border-white/[0.04]">
                {driverStints.map((stint, idx) => {
                  const compUpper = stint.compound.toUpperCase();
                  const compConf = COMPOUND_COLORS[compUpper] || { bg: "#71717a", text: "#fff", label: "?" };
                  const widthPct = Math.max(4, (stint.lap_count / totalLaps) * 100);

                  return (
                    <div
                      key={idx}
                      className="h-full rounded flex items-center justify-between px-2 text-[10px] font-mono font-bold transition-transform hover:scale-[1.02] shadow-sm relative group cursor-pointer"
                      style={{
                        width: `${widthPct}%`,
                        backgroundColor: compConf.bg,
                        color: compConf.text,
                      }}
                    >
                      <span>{compConf.label}</span>
                      <span className="opacity-80 text-[9px]">{stint.lap_count}L</span>

                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-[#141416] border border-white/10 text-white text-[10px] rounded px-2 py-1 shadow-lg whitespace-nowrap z-20">
                        Stint {stint.stint}: Lap {stint.start_lap} - {stint.end_lap} ({stint.lap_count} laps)
                        {stint.fresh_tyre && " - New"}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
