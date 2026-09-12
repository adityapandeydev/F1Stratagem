import React, { useState, useRef, useEffect } from "react";
import { ChevronDown, Check, Zap } from "lucide-react";

export interface CustomSelectOption {
  value: string | number;
  label: string;
  subLabel?: string;
  color?: string;
  isFastest?: boolean;
}

interface CustomSelectProps {
  options: CustomSelectOption[];
  value: string | number;
  onChange: (value: any) => void;
  placeholder?: string;
  className?: string;
  menuClassName?: string;
  pillMode?: boolean; // For compact driver/lap pills
  align?: "left" | "right";
  icon?: React.ReactNode;
}

export default function CustomSelect({
  options,
  value,
  onChange,
  placeholder = "Select...",
  className = "",
  menuClassName = "",
  pillMode = false,
  align = "left",
  icon,
}: CustomSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = options.find((o) => o.value === value);

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  const handleSelect = (val: string | number) => {
    onChange(val);
    setIsOpen(false);
  };

  return (
    <div ref={containerRef} className={`relative ${isOpen ? "z-50" : "z-10"} ${className}`}>
      {/* Clickable Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between gap-2.5 transition-all outline-none cursor-pointer select-none ${
          pillMode
            ? "px-3.5 py-2 rounded-xl bg-[#161622] hover:bg-[#1c1c2b] border border-white/[0.1] hover:border-white/[0.22] text-xs font-mono font-semibold text-neutral-200 shadow-sm"
            : "px-3.5 py-2.5 rounded-xl bg-[#15151c] hover:bg-[#1a1a24] border border-white/[0.1] hover:border-white/[0.22] focus:border-red-500 text-xs font-medium text-white shadow-md"
        } ${isOpen ? "border-red-500/80 ring-1 ring-red-500/40" : ""}`}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1 text-left">
          {icon && <span className="shrink-0">{icon}</span>}
          {selectedOption?.color && (
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
              style={{ backgroundColor: selectedOption.color }}
            />
          )}
          {selectedOption?.isFastest && (
            <Zap className="w-3 h-3 text-amber-400 shrink-0 fill-amber-400/20" />
          )}
          <span className="truncate text-left">
            {selectedOption ? selectedOption.label : placeholder}
          </span>
        </div>

        {/* Chevron with balanced 14px inset from right border */}
        <ChevronDown
          className={`w-3.5 h-3.5 text-neutral-400 shrink-0 ml-1.5 transition-transform duration-200 ${
            isOpen ? "rotate-180 text-white" : ""
          }`}
        />
      </button>

      {/* Custom Glassmorphic Popover Menu - Perfectly aligned with selector box */}
      {isOpen && (
        <div
          className={`absolute top-full mt-1.5 left-0 w-full min-w-full max-h-64 overflow-y-auto rounded-xl bg-[#0f0f16]/98 border border-white/[0.14] shadow-2xl backdrop-blur-2xl p-1.5 z-50 animate-in fade-in zoom-in-95 duration-150 ${menuClassName}`}
          style={{
            boxShadow: "0 20px 40px -15px rgba(0, 0, 0, 0.9), 0 0 1px 1px rgba(255, 255, 255, 0.1)",
          }}
        >
          {options.length === 0 ? (
            <div className="px-3 py-2 text-xs font-mono text-neutral-500 text-center">
              No options available
            </div>
          ) : (
            options.map((opt) => {
              const isSelected = opt.value === value;
              return (
                <div
                  key={opt.value}
                  onClick={() => handleSelect(opt.value)}
                  className={`flex items-center justify-between gap-2.5 px-3 py-2 rounded-lg text-xs font-mono cursor-pointer transition-all select-none ${
                    isSelected
                      ? "bg-red-500/15 text-white font-bold border border-red-500/25"
                      : "text-neutral-300 hover:text-white hover:bg-white/[0.08]"
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {opt.color && (
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
                        style={{ backgroundColor: opt.color }}
                      />
                    )}
                    {opt.isFastest && (
                      <Zap className="w-3 h-3 text-amber-400 shrink-0 fill-amber-400/20" />
                    )}
                    <span className="truncate">{opt.label}</span>
                  </div>

                  {opt.subLabel && (
                    <span className="text-[10px] text-neutral-400 font-normal shrink-0">
                      {opt.subLabel}
                    </span>
                  )}

                  {isSelected && (
                    <Check className="w-3.5 h-3.5 text-red-400 shrink-0" />
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
