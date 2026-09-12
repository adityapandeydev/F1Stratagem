import React from "react";
import { SessionResult } from "../../types";

interface ResultsTableProps {
  results: SessionResult[];
  isLoading?: boolean;
}

export default function ResultsTable({ results, isLoading }: ResultsTableProps) {
  const formatTime = (ms?: number | null) => {
    if (!ms) return "-";
    const mins = Math.floor(ms / 60000);
    const secs = ((ms % 60000) / 1000).toFixed(3);
    return `${mins}:${secs.padStart(6, "0")}`;
  };

  if (isLoading) {
    return (
      <div className="p-8 text-center text-text-tertiary">
        <div className="w-6 h-6 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-2" />
        Loading classification...
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="p-8 text-center text-text-tertiary">
        No results available for this session yet.
      </div>
    );
  }

  const leaderTime = results[0]?.race_time_ms || 0;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs font-mono">
        <thead>
          <tr className="border-b border-white/[0.08] text-neutral-400 uppercase tracking-wider text-[10px]">
            <th className="pb-3 pl-4 font-medium w-12">Pos</th>
            <th className="pb-3 font-medium">Driver</th>
            <th className="pb-3 font-medium">Team</th>
            <th className="pb-3 font-medium text-center w-16">Grid</th>
            <th className="pb-3 font-medium text-right">Time / Gap</th>
            <th className="pb-3 pr-4 font-medium text-right w-16">Pts</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {results.map((r, idx) => {
            const teamColor = r.team?.color || "#3b82f6";
            const pos = r.position || idx + 1;

            let timeGap = "-";
            if (r.race_time_ms) {
              if (pos === 1) {
                timeGap = formatTime(r.race_time_ms);
              } else if (leaderTime) {
                const diffSec = ((r.race_time_ms - leaderTime) / 1000).toFixed(3);
                timeGap = `+${diffSec}s`;
              }
            } else if (r.status && r.status !== "Finished") {
              timeGap = r.status;
            }

            return (
              <tr key={r.id || idx} className="hover:bg-white/[0.02] transition-colors group">
                <td className="py-3.5 pl-4 font-bold text-white">
                  <span
                    className={`inline-flex items-center justify-center w-6 h-6 rounded-md ${
                      pos === 1
                        ? "bg-amber-500/20 text-amber-300 font-extrabold"
                        : pos === 2
                        ? "bg-slate-300/20 text-slate-200"
                        : pos === 3
                        ? "bg-amber-700/20 text-amber-500"
                        : "text-neutral-400"
                    }`}
                  >
                    {pos}
                  </span>
                </td>
                <td className="py-3.5 font-sans">
                  <div className="flex items-center gap-2.5">
                    <span className="w-1 h-5 rounded-full" style={{ backgroundColor: teamColor }} />
                    <div>
                      <div className="font-semibold text-white text-sm flex items-center gap-1.5">
                        {r.driver?.full_name || `${r.driver?.first_name} ${r.driver?.last_name}`}
                        <span className="font-mono text-xs font-normal text-neutral-400">
                          #{r.driver?.driver_number || ""}
                        </span>
                      </div>
                      <div className="font-mono text-[11px] text-neutral-400 uppercase">
                        {r.driver?.abbreviation}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="py-3.5 font-sans text-neutral-300">{r.team?.name || "-"}</td>
                <td className="py-3.5 text-center text-neutral-400">{r.grid_position || "-"}</td>
                <td className="py-3.5 text-right font-medium text-white">{timeGap}</td>
                <td className="py-3.5 pr-4 text-right">
                  {r.points > 0 ? (
                    <span className="inline-block px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-bold">
                      +{r.points}
                    </span>
                  ) : (
                    <span className="text-neutral-500">0</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
