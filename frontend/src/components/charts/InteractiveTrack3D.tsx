import React, { useEffect, useRef, useState, useMemo } from "react";
import * as THREE from "three";
import { Play, Pause, RotateCcw, Compass, Zap, Layers } from "lucide-react";
import { TelemetryPoint, TelemetryDriverMeta } from "../../types";

interface InteractiveTrack3DProps {
  drivers: TelemetryDriverMeta[];
  telemetryStreams: TelemetryPoint[][];
  currentDistance?: number;
  onDistanceChange?: (dist: number) => void;
}

export default function InteractiveTrack3D({
  drivers,
  telemetryStreams,
  currentDistance,
  onDistanceChange,
}: InteractiveTrack3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [activeDistance, setActiveDistance] = useState(0);
  const [viewMode, setViewMode] = useState<"3D" | "Top">("3D");

  // Keep ref for current distance to prevent re-initializing Three.js on every scrubber tick
  const currentDistanceRef = useRef(currentDistance);
  currentDistanceRef.current = currentDistance;

  const stateRef = useRef({
    isPlaying: false,
    activeDistance: 0,
    maxDistance: 5412,
    playbackSpeed: 1,
  });

  stateRef.current.isPlaying = isPlaying;
  stateRef.current.activeDistance = activeDistance;
  stateRef.current.playbackSpeed = playbackSpeed;

  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    carMarkers: THREE.Group[];
    controlsState: {
      isDragging: boolean;
      prevX: number;
      prevY: number;
      theta: number;
      phi: number;
      radius: number;
      target: THREE.Vector3;
    };
    pointsMap: {
      d: number;
      pos: THREE.Vector3;
      dominantDriverIdx: number;
      speeds: number[];
    }[];
  } | null>(null);

  // Memoize driver abbreviations string to detect actual change
  const driversSignature = useMemo(
    () => drivers.map((d) => `${d.abbreviation}_${d.color}`).join("|"),
    [drivers]
  );

  const primaryTel = telemetryStreams[0] || [];

  useEffect(() => {
    if (!containerRef.current || primaryTel.length === 0) return;

    const width = containerRef.current.clientWidth || 800;
    const height = 480;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08080a);
    scene.fog = new THREE.FogExp2(0x08080a, 0.00035);

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 60000);
    camera.position.set(0, 1600, 2600);

    // 3. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(1200, 3000, 1500);
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x3b82f6, 0.4);
    fillLight.position.set(-1500, 1000, -1500);
    scene.add(fillLight);

    // 5. Grid Floor
    const grid = new THREE.GridHelper(7000, 70, 0x1c1c22, 0x101014);
    grid.position.y = -30;
    scene.add(grid);

    // 6. Coordinate Normalization
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of primaryTel) {
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
    const span = Math.max(maxX - minX, maxY - minY, 1);
    const scale = 1700 / span;

    const maxDist = primaryTel[primaryTel.length - 1]?.d || 5412;
    stateRef.current.maxDistance = maxDist;

    // Build 3D centerline points
    const curvePoints: THREE.Vector3[] = [];
    const pointsMap: {
      d: number;
      pos: THREE.Vector3;
      dominantDriverIdx: number;
      speeds: number[];
    }[] = [];

    const driverColors = drivers.map((d) => new THREE.Color(d.color || "#ffffff"));

    for (let i = 0; i < primaryTel.length; i++) {
      const p = primaryTel[i];
      // decimeters to Three.js coordinates
      const x = (p.x - midX) * scale;
      const y = (p.z - midZ) * scale * 2.2; // Elevation
      const z = -(p.y - midY) * scale;

      const pos = new THREE.Vector3(x, y, z);
      curvePoints.push(pos);

      // Determine fastest driver at this sample
      let bestSpd = -Infinity;
      let dominantIdx = 0;
      const speeds: number[] = [];

      for (let dIdx = 0; dIdx < telemetryStreams.length; dIdx++) {
        const dPt = telemetryStreams[dIdx]?.[i] || p;
        const spd = dPt.spd;
        speeds.push(spd);
        if (spd > bestSpd) {
          bestSpd = spd;
          dominantIdx = dIdx;
        }
      }

      pointsMap.push({
        d: p.d,
        pos,
        dominantDriverIdx: dominantIdx,
        speeds,
      });
    }

    // 7. Closed Curve for Track
    const curve = new THREE.CatmullRomCurve3(curvePoints, true, "centripetal");
    const numSubdivisions = 800;
    const smoothPoints = curve.getPoints(numSubdivisions);

    // 8. Extruded Asphalt Road Ribbon Geometry
    const roadWidth = 26;
    const ribbonPositions: number[] = [];
    const ribbonColors: number[] = [];
    const kerbLeftPositions: number[] = [];
    const kerbRightPositions: number[] = [];

    for (let i = 0; i < smoothPoints.length; i++) {
      const curr = smoothPoints[i];
      const next = smoothPoints[(i + 1) % smoothPoints.length];
      const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
      const up = new THREE.Vector3(0, 1, 0);
      const side = new THREE.Vector3().crossVectors(tangent, up).normalize();

      // Find closest sample point for dominance color
      const sampleIdx = Math.min(
        Math.floor((i / smoothPoints.length) * pointsMap.length),
        pointsMap.length - 1
      );
      const domColor = driverColors[pointsMap[sampleIdx]?.dominantDriverIdx || 0] || new THREE.Color(0x3b82f6);

      // Left & Right Road Edges
      const leftEdge = new THREE.Vector3().copy(curr).addScaledVector(side, -roadWidth / 2);
      const rightEdge = new THREE.Vector3().copy(curr).addScaledVector(side, roadWidth / 2);

      // Add to road ribbon
      ribbonPositions.push(leftEdge.x, leftEdge.y, leftEdge.z);
      ribbonPositions.push(rightEdge.x, rightEdge.y, rightEdge.z);

      // Subtle track surface with slight dominant glow
      const r = domColor.r * 0.4 + 0.08;
      const g = domColor.g * 0.4 + 0.08;
      const b = domColor.b * 0.4 + 0.08;

      ribbonColors.push(r, g, b);
      ribbonColors.push(r, g, b);

      // Outer kerbs
      const outerLeft = new THREE.Vector3().copy(leftEdge).addScaledVector(side, -3);
      const outerRight = new THREE.Vector3().copy(rightEdge).addScaledVector(side, 3);
      kerbLeftPositions.push(outerLeft.x, outerLeft.y + 0.5, outerLeft.z);
      kerbRightPositions.push(outerRight.x, outerRight.y + 0.5, outerRight.z);
    }

    // Build Triangle Indices for Road Surface
    const indices: number[] = [];
    for (let i = 0; i < smoothPoints.length; i++) {
      const p1 = i * 2;
      const p2 = i * 2 + 1;
      const p3 = ((i + 1) % smoothPoints.length) * 2;
      const p4 = ((i + 1) % smoothPoints.length) * 2 + 1;

      indices.push(p1, p2, p3);
      indices.push(p2, p4, p3);
    }

    const roadGeo = new THREE.BufferGeometry();
    roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(ribbonPositions, 3));
    roadGeo.setAttribute("color", new THREE.Float32BufferAttribute(ribbonColors, 3));
    roadGeo.setIndex(indices);
    roadGeo.computeVertexNormals();

    const roadMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.7,
      metalness: 0.2,
      side: THREE.DoubleSide,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    scene.add(roadMesh);

    // 9. Glowing Centerline Ribbon (True Dominance Trail)
    const linePositions: number[] = [];
    const lineColors: number[] = [];

    for (let i = 0; i < smoothPoints.length; i++) {
      const curr = smoothPoints[i];
      const sampleIdx = Math.min(
        Math.floor((i / smoothPoints.length) * pointsMap.length),
        pointsMap.length - 1
      );
      const domColor = driverColors[pointsMap[sampleIdx]?.dominantDriverIdx || 0] || new THREE.Color(0xffffff);

      linePositions.push(curr.x, curr.y + 1.2, curr.z);
      lineColors.push(domColor.r, domColor.g, domColor.b);
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeo.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));
    const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 3 });
    const centerline = new THREE.LineLoop(lineGeo, lineMat);
    scene.add(centerline);

    // 10. Start / Finish Gate Marker
    const startPt = smoothPoints[0];
    const startNext = smoothPoints[1];
    const startTan = new THREE.Vector3().subVectors(startNext, startPt).normalize();
    const startSide = new THREE.Vector3().crossVectors(startTan, new THREE.Vector3(0, 1, 0)).normalize();

    const finishLineGeo = new THREE.BoxGeometry(roadWidth + 4, 3, 4);
    const finishLineMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.6,
    });
    const finishGate = new THREE.Mesh(finishLineGeo, finishLineMat);
    finishGate.position.copy(startPt);
    finishGate.position.y += 1.5;
    finishGate.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), startSide);
    scene.add(finishGate);

    // 11. Car Markers for Each Driver (up to 4)
    const carMarkers: THREE.Group[] = [];
    const lateralOffsets = [-9, 9, -15, 15]; // stagger cars across track width

    drivers.forEach((drv, idx) => {
      const group = new THREE.Group();
      const col = new THREE.Color(drv.color || "#ffffff");

      // Core Car Orb
      const sphereGeo = new THREE.SphereGeometry(18, 20, 20);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.85,
        roughness: 0.1,
        metalness: 0.8,
      });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      group.add(sphere);

      // Outer Halo Ring
      const ringGeo = new THREE.RingGeometry(24, 28, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: col,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      // Drop Pin / Vertical Stalk
      const stalkGeo = new THREE.CylinderGeometry(1.5, 1.5, 18, 8);
      const stalkMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 });
      const stalk = new THREE.Mesh(stalkGeo, stalkMat);
      stalk.position.y = -9;
      group.add(stalk);

      group.position.copy(startPt);
      group.position.y += 18;
      scene.add(group);
      carMarkers.push(group);
    });

    // 12. Orbit Camera Controls State
    const controlsState = {
      isDragging: false,
      prevX: 0,
      prevY: 0,
      theta: Math.PI / 4,
      phi: Math.PI / 3.4,
      radius: 2600,
      target: new THREE.Vector3(0, 0, 0),
    };

    const updateCameraPos = () => {
      const { theta, phi, radius, target } = controlsState;
      camera.position.x = target.x + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.y = target.y + radius * Math.cos(phi);
      camera.position.z = target.z + radius * Math.sin(phi) * Math.cos(theta);
      camera.lookAt(target);
    };
    updateCameraPos();

    // DOM Mouse Listeners for Orbit
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
      controlsState.phi = Math.max(0.05, Math.min(Math.PI / 2.05, controlsState.phi - dy * 0.005));
      updateCameraPos();
    };
    const onMouseUp = () => {
      controlsState.isDragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      controlsState.radius = Math.max(500, Math.min(6500, controlsState.radius + e.deltaY * 1.5));
      updateCameraPos();
    };

    dom.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    dom.addEventListener("wheel", onWheel, { passive: false });

    sceneRef.current = {
      scene,
      camera,
      renderer,
      carMarkers,
      controlsState,
      pointsMap,
    };

    // 13. Animation Loop
    let animId: number;
    let lastTime = performance.now();

    const animate = (time: number) => {
      animId = requestAnimationFrame(animate);
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      if (stateRef.current.isPlaying) {
        const speedKmh = 270 * stateRef.current.playbackSpeed;
        const distDelta = ((speedKmh * 1000) / 3600) * dt;
        let newDist = stateRef.current.activeDistance + distDelta;
        if (newDist > stateRef.current.maxDistance) {
          newDist = 0;
        }
        setActiveDistance(newDist);
        if (onDistanceChange) onDistanceChange(newDist);
      }

      const currentDist =
        currentDistanceRef.current !== undefined
          ? currentDistanceRef.current
          : stateRef.current.activeDistance;

      // Update position of all car markers
      if (pointsMap.length > 0) {
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
          carMarkers.forEach((group, idx) => {
            const latOff = lateralOffsets[idx] || 0;
            const targetPos = new THREE.Vector3().copy(pt.pos);
            targetPos.y += 16;
            targetPos.x += latOff;
            group.position.lerp(targetPos, 0.35);
          });
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
      roadGeo.dispose();
      roadMat.dispose();
      lineGeo.dispose();
      lineMat.dispose();
    };
  }, [driversSignature, primaryTel.length]);

  const handleViewToggle = () => {
    if (!sceneRef.current) return;
    const { controlsState, camera } = sceneRef.current;
    if (viewMode === "3D") {
      setViewMode("Top");
      controlsState.phi = 0.02;
      controlsState.theta = 0;
      controlsState.radius = 3200;
    } else {
      setViewMode("3D");
      controlsState.phi = Math.PI / 3.4;
      controlsState.theta = Math.PI / 4;
      controlsState.radius = 2600;
    }
    const { theta, phi, radius, target } = controlsState;
    camera.position.x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    camera.position.y = target.y + radius * Math.cos(phi);
    camera.position.z = target.z + radius * Math.sin(phi) * Math.cos(theta);
    camera.lookAt(target);
  };

  const handleResetCamera = () => {
    if (!sceneRef.current) return;
    const { controlsState, camera } = sceneRef.current;
    controlsState.phi = Math.PI / 3.4;
    controlsState.theta = Math.PI / 4;
    controlsState.radius = 2600;
    controlsState.target.set(0, 0, 0);
    const { theta, phi, radius, target } = controlsState;
    camera.position.x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    camera.position.y = target.y + radius * Math.cos(phi);
    camera.position.z = target.z + radius * Math.sin(phi) * Math.cos(theta);
    camera.lookAt(target);
    setViewMode("3D");
  };

  const currentDist = currentDistance !== undefined ? currentDistance : activeDistance;
  const maxD = primaryTel.length > 0 ? primaryTel[primaryTel.length - 1].d : 5412;

  // Active speeds at current distance
  const currentSpeeds = useMemo(() => {
    if (!sceneRef.current || sceneRef.current.pointsMap.length === 0) return [];
    const pts = sceneRef.current.pointsMap;
    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const diff = Math.abs(pts[i].d - currentDist);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    return pts[closestIdx]?.speeds || [];
  }, [currentDist]);

  return (
    <div className="relative rounded-2xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl overflow-hidden shadow-2xl">
      {/* Top HUD Overlay */}
      <div className="absolute top-4 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Track Title & Dominance Legend */}
        <div className="flex items-center gap-2.5 pointer-events-auto bg-[#141416]/85 backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-white/[0.08] shadow-lg">
          <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-neutral-300">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            FastF1 GPS Circuit (Sakhir)
          </span>
          <span className="text-white/20">|</span>
          <div className="flex items-center gap-3">
            {drivers.map((d, idx) => (
              <span key={d.abbreviation} className="flex items-center gap-1.5 text-xs font-mono font-medium">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
                <span style={{ color: d.color }}>{d.abbreviation}</span>
                {currentSpeeds[idx] !== undefined && (
                  <span className="text-[11px] text-neutral-400 font-normal">
                    {Math.round(currentSpeeds[idx])} km/h
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={handleResetCamera}
            className="p-1.5 rounded-xl text-xs font-medium bg-[#141416]/85 hover:bg-[#1f1f23] text-neutral-400 hover:text-white border border-white/[0.08] transition-all shadow-md"
            title="Reset Camera View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleViewToggle}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-[#141416]/85 hover:bg-[#1f1f23] text-neutral-300 hover:text-white border border-white/[0.08] transition-all shadow-md"
          >
            <Compass className="w-3.5 h-3.5" />
            {viewMode} View
          </button>
        </div>
      </div>

      {/* 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full h-[480px] cursor-grab active:cursor-grabbing" />

      {/* Bottom Scrubber & Playback Controls */}
      <div className="p-4 bg-[#0d0d0f]/95 border-t border-white/[0.06] flex flex-col md:flex-row items-center gap-4">
        <div className="flex items-center gap-2 shrink-0">
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
            className="w-9 h-9 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-neutral-400 hover:text-white flex items-center justify-center transition-all"
            title="Reset to Start/Finish"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-1 ml-1 bg-white/[0.04] p-1 rounded-xl border border-white/[0.06]">
            {[1, 2, 4].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaybackSpeed(spd)}
                className={`px-2 py-0.5 text-xs rounded-lg font-mono transition-all ${
                  playbackSpeed === spd
                    ? "bg-white/20 text-white font-bold"
                    : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>

        {/* Scrub Slider */}
        <div className="flex-1 w-full flex items-center gap-3">
          <span className="text-xs font-mono text-neutral-400 w-16 text-right shrink-0">
            {Math.round(currentDist).toLocaleString()}m
          </span>
          <input
            type="range"
            min={0}
            max={maxD || 5412}
            step={5}
            value={currentDist}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setActiveDistance(val);
              if (onDistanceChange) onDistanceChange(val);
            }}
            className="flex-1 accent-red-500 h-1.5 bg-white/10 rounded-lg cursor-pointer"
          />
          <span className="text-xs font-mono text-neutral-500 w-16 shrink-0">
            {Math.round(maxD).toLocaleString()}m
          </span>
        </div>
      </div>
    </div>
  );
}
