import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Play, Pause, RotateCcw, Eye, Compass } from "lucide-react";
import { TelemetryPoint } from "../../types";

interface DriverInfo {
  abbreviation: string;
  name?: string;
  color: string;
}

interface InteractiveTrack3DProps {
  telemetry1: TelemetryPoint[];
  telemetry2: TelemetryPoint[];
  driver1: DriverInfo;
  driver2: DriverInfo;
  currentDistance?: number;
  onDistanceChange?: (dist: number) => void;
}

export default function InteractiveTrack3D({
  telemetry1,
  telemetry2,
  driver1,
  driver2,
  currentDistance,
  onDistanceChange,
}: InteractiveTrack3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [activeDistance, setActiveDistance] = useState(0);
  const [viewMode, setViewMode] = useState<"3D" | "Top">("3D");

  // Keep ref to latest state for animation loop
  const stateRef = useRef({
    isPlaying: false,
    activeDistance: 0,
    maxDistance: 1000,
    playbackSpeed: 1,
  });

  stateRef.current.isPlaying = isPlaying;
  stateRef.current.activeDistance = activeDistance;
  stateRef.current.playbackSpeed = playbackSpeed;

  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    marker1: THREE.Mesh;
    marker2: THREE.Mesh;
    trackLine: THREE.Line;
    trackRibbon: THREE.Mesh;
    controlsState: { isDragging: boolean; prevX: number; prevY: number; theta: number; phi: number; radius: number; target: THREE.Vector3 };
    pointsMap: { d: number; pos: THREE.Vector3; faster: "d1" | "d2" | "equal"; spd1: number; spd2: number }[];
  } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const width = containerRef.current.clientWidth || 800;
    const height = 460;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0b);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 50000);
    camera.position.set(0, 1500, 2500);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(renderer.domElement);

    // Subtle Grid & Lighting
    const grid = new THREE.GridHelper(5000, 50, 0x1f1f23, 0x141416);
    grid.position.y = -20;
    scene.add(grid);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(1000, 2000, 1000);
    scene.add(dirLight);

    // Process coordinates from telemetry
    const tel = telemetry1.length > 0 ? telemetry1 : telemetry2;
    if (tel.length === 0) return;

    // Center coordinates
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of tel) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const midZ = (minZ + maxZ) / 2;
    const scale = 0.25;

    const pointsMap: { d: number; pos: THREE.Vector3; faster: "d1" | "d2" | "equal"; spd1: number; spd2: number }[] = [];
    const maxDist = tel[tel.length - 1]?.d || 5000;
    stateRef.current.maxDistance = maxDist;

    // Track path vertices and colors
    const colors: number[] = [];
    const color1 = new THREE.Color(driver1.color || "#3b82f6");
    const color2 = new THREE.Color(driver2.color || "#ef4444");
    const colorEqual = new THREE.Color(0x52525b);

    const curvePoints: THREE.Vector3[] = [];

    for (let i = 0; i < tel.length; i++) {
      const p1 = telemetry1[i] || tel[i];
      const p2 = telemetry2[i] || tel[i];

      // Elevation comes from Z (or Y depending on coordinate system)
      const x = (p1.x - midX) * scale;
      const y = (p1.z - midZ) * scale * 1.5; // elevate for 3D visibility
      const z = -(p1.y - midY) * scale;

      const pos = new THREE.Vector3(x, y, z);
      curvePoints.push(pos);

      let faster: "d1" | "d2" | "equal" = "equal";
      let segmentColor = colorEqual;

      if (p1 && p2) {
        if (p1.spd > p2.spd + 2) {
          faster = "d1";
          segmentColor = color1;
        } else if (p2.spd > p1.spd + 2) {
          faster = "d2";
          segmentColor = color2;
        }
      }

      colors.push(segmentColor.r, segmentColor.g, segmentColor.b);

      pointsMap.push({
        d: p1.d,
        pos,
        faster,
        spd1: p1.spd,
        spd2: p2 ? p2.spd : p1.spd,
      });
    }

    // Create 3D track ribbon using CatmullRomCurve3
    const curve = new THREE.CatmullRomCurve3(curvePoints, false);
    const tubeGeo = new THREE.TubeGeometry(curve, Math.min(curvePoints.length, 600), 12, 8, false);

    // Apply vertex colors to tube
    const tubeColors: number[] = [];
    const posAttr = tubeGeo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const idx = Math.min(Math.floor((i / posAttr.count) * colors.length / 3) * 3, colors.length - 3);
      tubeColors.push(colors[idx] || 0.4, colors[idx + 1] || 0.4, colors[idx + 2] || 0.4);
    }
    tubeGeo.setAttribute("color", new THREE.Float32BufferAttribute(tubeColors, 3));

    const tubeMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.3,
      metalness: 0.2,
    });
    const trackRibbon = new THREE.Mesh(tubeGeo, tubeMat);
    scene.add(trackRibbon);

    // Track centerline outline
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curvePoints);
    lineGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    const trackLine = new THREE.Line(lineGeo, lineMat);
    scene.add(trackLine);

    // Driver 1 Glowing Orb Marker
    const marker1Geo = new THREE.SphereGeometry(22, 16, 16);
    const marker1Mat = new THREE.MeshStandardMaterial({
      color: color1,
      emissive: color1,
      emissiveIntensity: 0.8,
    });
    const marker1 = new THREE.Mesh(marker1Geo, marker1Mat);
    scene.add(marker1);

    // Driver 2 Glowing Orb Marker
    const marker2Geo = new THREE.SphereGeometry(22, 16, 16);
    const marker2Mat = new THREE.MeshStandardMaterial({
      color: color2,
      emissive: color2,
      emissiveIntensity: 0.8,
    });
    const marker2 = new THREE.Mesh(marker2Geo, marker2Mat);
    scene.add(marker2);

    // Position markers initially
    if (pointsMap.length > 0) {
      marker1.position.copy(pointsMap[0].pos);
      marker2.position.copy(pointsMap[0].pos);
    }

    // Camera orbit controls
    const controlsState = {
      isDragging: false,
      prevX: 0,
      prevY: 0,
      theta: Math.PI / 4,
      phi: Math.PI / 3.5,
      radius: 2600,
      target: new THREE.Vector3(0, 0, 0),
    };

    const updateCameraPos = () => {
      camera.position.x = controlsState.target.x + controlsState.radius * Math.sin(controlsState.phi) * Math.sin(controlsState.theta);
      camera.position.y = controlsState.target.y + controlsState.radius * Math.cos(controlsState.phi);
      camera.position.z = controlsState.target.z + controlsState.radius * Math.sin(controlsState.phi) * Math.cos(controlsState.theta);
      camera.lookAt(controlsState.target);
    };
    updateCameraPos();

    // Mouse handlers
    const dom = renderer.domElement;
    const onMouseDown = (e: MouseEvent) => {
      controlsState.isDragging = true;
      controlsState.prevX = e.clientX;
      controlsState.prevY = e.clientY;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!controlsState.isDragging) return;
      const dx = e.clientX - controlsState.prevX;
      const dy = e.clientY - controlsState.prevY;
      controlsState.prevX = e.clientX;
      controlsState.prevY = e.clientY;

      controlsState.theta -= dx * 0.005;
      controlsState.phi = Math.max(0.1, Math.min(Math.PI / 2.1, controlsState.phi - dy * 0.005));
      updateCameraPos();
    };
    const onMouseUp = () => {
      controlsState.isDragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      controlsState.radius = Math.max(500, Math.min(6000, controlsState.radius + e.deltaY * 1.5));
      updateCameraPos();
    };

    dom.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    dom.addEventListener("wheel", onWheel);

    sceneRef.current = {
      scene,
      camera,
      renderer,
      marker1,
      marker2,
      trackLine,
      trackRibbon,
      controlsState,
      pointsMap,
    };

    // Animation Loop
    let animId: number;
    let lastTime = performance.now();

    const animate = (time: number) => {
      animId = requestAnimationFrame(animate);
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      if (stateRef.current.isPlaying) {
        const speedKmh = 250 * stateRef.current.playbackSpeed; // simulation speed
        const distDelta = (speedKmh * 1000 / 3600) * dt;
        let newDist = stateRef.current.activeDistance + distDelta;
        if (newDist > stateRef.current.maxDistance) {
          newDist = 0;
        }
        setActiveDistance(newDist);
        if (onDistanceChange) onDistanceChange(newDist);
      }

      // Update marker positions
      const currentDist = currentDistance !== undefined ? currentDistance : stateRef.current.activeDistance;
      if (pointsMap.length > 0) {
        // Find closest index
        let closestIdx = 0;
        let minDiff = Infinity;
        for (let i = 0; i < pointsMap.length; i++) {
          const diff = Math.abs(pointsMap[i].d - currentDist);
          if (diff < minDiff) {
            minDiff = diff;
            closestIdx = i;
          }
        }
        const pt = pointsMap[closestIdx];
        if (pt) {
          marker1.position.lerp(pt.pos, 0.3);
          marker2.position.lerp(pt.pos, 0.3);
          // offset slightly so both visible
          marker1.position.y += 6;
          marker2.position.y += 6;
          marker1.position.x -= 8;
          marker2.position.x += 8;
        }
      }

      renderer.render(scene, camera);
    };
    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      dom.removeEventListener("wheel", onWheel);
      renderer.dispose();
      tubeGeo.dispose();
      tubeMat.dispose();
    };
  }, [telemetry1, telemetry2, driver1, driver2]);

  // View mode switcher
  const handleViewToggle = () => {
    if (!sceneRef.current) return;
    const { controlsState } = sceneRef.current;
    if (viewMode === "3D") {
      setViewMode("Top");
      controlsState.phi = 0.05;
      controlsState.theta = 0;
    } else {
      setViewMode("3D");
      controlsState.phi = Math.PI / 3.5;
      controlsState.theta = Math.PI / 4;
    }
  };

  const currentDist = currentDistance !== undefined ? currentDistance : activeDistance;
  const maxD = telemetry1.length > 0 ? telemetry1[telemetry1.length - 1].d : 5000;

  return (
    <div className="relative rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl overflow-hidden shadow-2xl">
      {/* Header Overlay */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-3 pointer-events-auto bg-[#141416]/80 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/[0.08]">
          <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-text-secondary">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            3D Track Elevation & Speed Fluid
          </span>
          <span className="text-white/20">|</span>
          <div className="flex items-center gap-2 text-xs">
            <span className="flex items-center gap-1 font-mono font-medium" style={{ color: driver1.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: driver1.color }} />
              {driver1.abbreviation} Faster
            </span>
            <span className="flex items-center gap-1 font-mono font-medium" style={{ color: driver2.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: driver2.color }} />
              {driver2.abbreviation} Faster
            </span>
          </div>
        </div>

        {/* View Toggle */}
        <button
          onClick={handleViewToggle}
          className="pointer-events-auto flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-[#141416]/80 hover:bg-[#1f1f23] text-text-secondary hover:text-text-primary border border-white/[0.08] transition-all"
        >
          <Compass className="w-3.5 h-3.5" />
          {viewMode} View
        </button>
      </div>

      {/* 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full h-[460px] cursor-grab active:cursor-grabbing" />

      {/* Playback & Scrubber Controls Bar */}
      <div className="p-4 bg-[#0d0d0f]/90 border-t border-white/[0.06] flex flex-col md:flex-row items-center gap-4">
        {/* Play/Pause & Reset */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-9 h-9 rounded-full bg-white text-black hover:bg-neutral-200 flex items-center justify-center transition-transform active:scale-95 shadow-md"
            title={isPlaying ? "Pause" : "Play Telemetry Flow"}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
          </button>
          <button
            onClick={() => {
              setActiveDistance(0);
              if (onDistanceChange) onDistanceChange(0);
            }}
            className="w-9 h-9 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-text-secondary hover:text-text-primary flex items-center justify-center transition-all"
            title="Reset to Start"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-1 ml-2 bg-white/[0.04] p-1 rounded-lg border border-white/[0.06]">
            {[1, 2, 4].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaybackSpeed(spd)}
                className={`px-2 py-0.5 text-xs rounded font-mono transition-all ${
                  playbackSpeed === spd ? "bg-white/20 text-white font-bold" : "text-text-tertiary hover:text-white"
                }`}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>

        {/* Scrubber */}
        <div className="flex-1 w-full flex items-center gap-3">
          <span className="text-xs font-mono text-text-tertiary w-14 text-right">
            {Math.round(currentDist)}m
          </span>
          <input
            type="range"
            min={0}
            max={maxD || 5000}
            step={5}
            value={currentDist}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setActiveDistance(val);
              if (onDistanceChange) onDistanceChange(val);
            }}
            className="flex-1 accent-white h-1.5 bg-white/10 rounded-lg cursor-pointer"
          />
          <span className="text-xs font-mono text-text-tertiary w-14">
            {Math.round(maxD)}m
          </span>
        </div>
      </div>
    </div>
  );
}
