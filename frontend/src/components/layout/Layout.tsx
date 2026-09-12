import React, { useState } from "react";
import { Outlet, NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Calendar,
  Zap,
  GitCompare,
  MapPin,
  Users,
  Shield,
  BarChart3,
  ChevronLeft,
  Flame,
} from "lucide-react";

interface NavGroup {
  section: string;
  items: {
    path: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
  }[];
}

const navGroups: NavGroup[] = [
  {
    section: "EXPLORE",
    items: [
      { path: "/", label: "Dashboard", icon: LayoutDashboard },
      { path: "/session/1", label: "Race Sessions", icon: Calendar },
    ],
  },
  {
    section: "ANALYSIS",
    items: [
      { path: "/compare", label: "Telemetry", icon: GitCompare },
      { path: "/circuits", label: "Circuits", icon: MapPin },
    ],
  },
  {
    section: "DATABASE",
    items: [
      { path: "/drivers", label: "Drivers", icon: Users },
      { path: "/teams", label: "Constructors", icon: Shield },
      { path: "/standings", label: "Standings", icon: BarChart3 },
    ],
  },
];

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-[#09090b] text-[#ededed]">
      {/* Hover-Expand Floating Sidebar Rail */}
      <aside
        className="
          group flex flex-col border-r border-white/[0.07]
          bg-[#0e0e12]/95 backdrop-blur-2xl
          w-16 hover:w-64
          transition-all duration-300 ease-out z-30 shrink-0
          hover:shadow-2xl hover:shadow-black/70 overflow-x-hidden
        "
      >
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-3.5 h-16 border-b border-white/[0.06] shrink-0 overflow-hidden">
          <div className="w-9 h-9 rounded-xl bg-red-600 flex items-center justify-center text-white text-xs font-black shadow-lg shadow-red-600/30 shrink-0">
            <Flame className="w-5 h-5 fill-white" />
          </div>
          <div className="leading-tight opacity-0 group-hover:opacity-100 transition-opacity duration-200 delay-75 whitespace-nowrap">
            <span className="text-sm font-black tracking-tight text-white">
              F1<span className="text-neutral-400 font-medium">Stratagem</span>
            </span>
            <span className="block text-[9px] font-mono text-neutral-500 uppercase tracking-widest">
              Telemetry Studio
            </span>
          </div>
        </div>

        {/* Navigation Groups */}
        <nav className="flex-1 overflow-y-auto py-5 px-2.5 space-y-6 overflow-x-hidden">
          {navGroups.map((group) => (
            <div key={group.section}>
              <p className="px-3 mb-2 text-[10px] font-mono font-bold tracking-wider text-neutral-500 uppercase opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
                {group.section}
              </p>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      className={({ isActive }) => `
                        flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium
                        transition-all duration-150 whitespace-nowrap
                        ${
                          isActive
                            ? "bg-white/10 text-white shadow-sm border border-white/[0.09]"
                            : "text-neutral-400 hover:text-white hover:bg-white/[0.04]"
                        }
                      `}
                      title={item.label}
                    >
                      <Icon className="w-4 h-4 shrink-0" />
                      <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 delay-75">
                        {item.label}
                      </span>
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom Status Dot */}
        <div className="p-3 border-t border-white/[0.06] flex items-center gap-2.5 overflow-hidden">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50 shrink-0 ml-1.5" />
          <span className="text-[10px] font-mono text-neutral-400 opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
            FastF1 Engine Ready
          </span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto relative bg-[#09090b]">
        <div className="max-w-[1500px] mx-auto px-6 lg:px-10 py-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
