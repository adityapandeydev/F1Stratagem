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
      { path: "/compare", label: "Telemetry & 3D", icon: GitCompare },
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
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0a0b] text-[#ededed]">
      {/* Sidebar */}
      <aside
        className={`
          flex flex-col border-r border-white/[0.06]
          bg-[#0d0d0f]/90 backdrop-blur-2xl
          transition-all duration-300 ease-in-out z-20 shrink-0
          ${collapsed ? "w-16" : "w-64"}
        `}
      >
        {/* Brand Header */}
        <div className="flex items-center justify-between px-5 h-16 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-red-600 flex items-center justify-center text-white text-xs font-black shadow-lg shadow-red-600/30">
              <Flame className="w-4 h-4 fill-white" />
            </div>
            {!collapsed && (
              <div className="leading-tight">
                <span className="text-sm font-extrabold tracking-tight text-white">
                  F1<span className="text-neutral-400 font-medium">Stratagem</span>
                </span>
                <span className="block text-[9px] font-mono text-neutral-500 uppercase tracking-wider">
                  Telemetry Engine
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Groups */}
        <nav className="flex-1 overflow-y-auto py-5 px-3 space-y-6">
          {navGroups.map((group) => (
            <div key={group.section}>
              {!collapsed && (
                <p className="px-3 mb-2 text-[10px] font-mono font-bold tracking-wider text-neutral-500 uppercase">
                  {group.section}
                </p>
              )}
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      className={({ isActive }) => `
                        flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium
                        transition-all duration-150
                        ${
                          isActive
                            ? "bg-white/10 text-white shadow-sm border border-white/[0.08]"
                            : "text-neutral-400 hover:text-white hover:bg-white/[0.04]"
                        }
                      `}
                      title={collapsed ? item.label : undefined}
                    >
                      <Icon className="w-4 h-4 shrink-0" />
                      {!collapsed && <span>{item.label}</span>}
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Sidebar Footer / Collapse toggle */}
        <div className="p-3 border-t border-white/[0.06]">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-xl text-neutral-400 hover:text-white hover:bg-white/[0.04] transition-all text-xs"
          >
            <ChevronLeft
              className={`w-4 h-4 transition-transform duration-300 ${
                collapsed ? "rotate-180" : ""
              }`}
            />
            {!collapsed && <span className="font-mono text-[11px]">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto relative">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
