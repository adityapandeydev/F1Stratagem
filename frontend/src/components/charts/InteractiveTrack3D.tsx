import React, { useEffect, useRef, useState, useMemo } from "react";
import * as THREE from "three";
import {
  Play,
  Pause,
  RotateCcw,
  Compass,
  Zap,
  Gauge,
  Activity,
  Layers,
  Sparkles,
} from "lucide-react";
import { TelemetryPoint, TelemetryDriverMeta } from "../../types";

interface InteractiveTrack3DProps {
  drivers: TelemetryDriverMeta[];
  telemetryStreams: TelemetryPoint[][];
  timeDelta?: number[];
  currentDistance?: number;
  onDistanceChange?: (dist: number) => void;
}

export default function InteractiveTrack3D({
  drivers,
  telemetryStreams,
  timeDelta = [],
  currentDistance,
  onDistanceChange,
}: InteractiveTrack3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [activeDistance, setActiveDistance] = useState(0);
  const [viewMode, setViewMode] = useState<"3D" | "Top">("3D");
  const [autoRotate, setAutoRotate] = useState(false);

  // Keep ref for current distance to prevent re-initializing Three.js on every scrubber tick
  const currentDistanceRef = useRef(currentDistance);
  currentDistanceRef.current = currentDistance;

  const stateRef = useRef({
    isPlaying: false,
    autoRotate: false,
    activeDistance: 0,
    maxDistance: 5412,
    playbackSpeed: 1,
  });

  stateRef.current.isPlaying = isPlaying;
  stateRef.current.autoRotate = autoRotate;
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
      gears: number[];
      throttles: number[];
      brakes: number[];
      deltaSeconds: number;
    }[];
  } | null>(null);

  // Memoize driver signature to detect changes in colors or abbreviations
  const driversSignature = useMemo(
    () => drivers.map((d) => `${d.abbreviation}_${d.color}`).join("|"),
    [drivers]
  );

  const primaryTel = telemetryStreams[0] || [];

  useEffect(() => {
    if (!containerRef.current || primaryTel.length === 0) return;

    const width = containerRef.current.clientWidth || 850;
    const height = 500;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070709);
    scene.fog = new THREE.FogExp2(0x070709, 0.00028);

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(42, width / height, 1, 70000);
    camera.position.set(0, 1600, 2600);

    // 3. WebGL Renderer with High-Precision Shadow Maps
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(renderer.domElement);

    // 4. TrackSims Studio Lighting Environment
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const studioKeyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    studioKeyLight.position.set(1500, 3400, 1800);
    studioKeyLight.castShadow = true;
    scene.add(studioKeyLight);

    const rimLight = new THREE.DirectionalLight(0x60a5fa, 0.45);
    rimLight.position.set(-2000, 1200, -2000);
    scene.add(rimLight);

    const warmFillLight = new THREE.DirectionalLight(0xffeedd, 0.35);
    warmFillLight.position.set(2000, 900, -1000);
    scene.add(warmFillLight);

    // 5. Studio Radial Contact Shadow Ground Disc
    const groundCanvas = document.createElement("canvas");
    groundCanvas.width = 512;
    groundCanvas.height = 512;
    const gCtx = groundCanvas.getContext("2d");
    if (gCtx) {
      const grad = gCtx.createRadialGradient(256, 256, 10, 256, 256, 250);
      grad.addColorStop(0, "rgba(22, 22, 28, 0.9)");
      grad.addColorStop(0.5, "rgba(14, 14, 18, 0.5)");
      grad.addColorStop(1, "rgba(7, 7, 9, 0)");
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, 512, 512);
    }
    const groundTex = new THREE.CanvasTexture(groundCanvas);
    const groundGeo = new THREE.PlaneGeometry(5600, 5600);
    const groundMat = new THREE.MeshBasicMaterial({
      map: groundTex,
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.position.y = -35;
    scene.add(groundMesh);

    // Subtle Grid overlay for technical precision
    const grid = new THREE.GridHelper(6500, 65, 0x18181d, 0x0e0e12);
    grid.position.y = -34;
    scene.add(grid);

    // 6. Coordinate Normalization & Decimeter Scaling
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

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
    const scale = 1750 / span;

    const maxDist = primaryTel[primaryTel.length - 1]?.d || 5412;
    stateRef.current.maxDistance = maxDist;

    // Build 3D centerline points & Dominance Analysis
    const curvePoints: THREE.Vector3[] = [];
    const pointsMap: {
      d: number;
      pos: THREE.Vector3;
      dominantDriverIdx: number;
      speeds: number[];
      gears: number[];
      throttles: number[];
      brakes: number[];
      deltaSeconds: number;
    }[] = [];

    const driverColors = drivers.map((d) => new THREE.Color(d.color || "#ffffff"));

    for (let i = 0; i < primaryTel.length; i++) {
      const p = primaryTel[i];
      // Decimeters to Three.js coordinates
      const x = (p.x - midX) * scale;
      const y = (p.z - midZ) * scale * 2.4; // Elevation
      const z = -(p.y - midY) * scale;

      const pos = new THREE.Vector3(x, y, z);
      curvePoints.push(pos);

      // Determine fastest driver at this sample
      let bestSpd = -Infinity;
      let dominantIdx = 0;
      const speeds: number[] = [];
      const gears: number[] = [];
      const throttles: number[] = [];
      const brakes: number[] = [];

      for (let dIdx = 0; dIdx < telemetryStreams.length; dIdx++) {
        const dPt = telemetryStreams[dIdx]?.[i] || p;
        const spd = dPt.spd;
        speeds.push(spd);
        gears.push(dPt.gear || 7);
        throttles.push(dPt.thr || 0);
        brakes.push(dPt.brk || 0);

        if (spd > bestSpd) {
          bestSpd = spd;
          dominantIdx = dIdx;
        }
      }

      const deltaSec = timeDelta && timeDelta[i] !== undefined ? timeDelta[i] : 0;

      pointsMap.push({
        d: p.d,
        pos,
        dominantDriverIdx: dominantIdx,
        speeds,
        gears,
        throttles,
        brakes,
        deltaSeconds: deltaSec,
      });
    }

    // 7. Smooth Catmull-Rom Spline
    const curve = new THREE.CatmullRomCurve3(curvePoints, true, "centripetal");
    const numSubdivisions = 850;
    const smoothPoints = curve.getPoints(numSubdivisions);

    // 8. TrackSims Extruded Asphalt Road Ribbon
    const roadWidth = 28;
    const roadPositions: number[] = [];
    const roadColors: number[] = [];
    const kerbLeftPositions: number[] = [];
    const kerbRightPositions: number[] = [];
    const dominanceOverlayPositions: number[] = [];
    const dominanceOverlayColors: number[] = [];

    for (let i = 0; i < smoothPoints.length; i++) {
      const curr = smoothPoints[i];
      const next = smoothPoints[(i + 1) % smoothPoints.length];
      const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
      const up = new THREE.Vector3(0, 1, 0);
      const side = new THREE.Vector3().crossVectors(tangent, up).normalize();

      const sampleIdx = Math.min(
        Math.floor((i / smoothPoints.length) * pointsMap.length),
        pointsMap.length - 1
      );
      const domColor = driverColors[pointsMap[sampleIdx]?.dominantDriverIdx || 0] || new THREE.Color(0x3b82f6);

      // Base Asphalt road edges
      const leftEdge = new THREE.Vector3().copy(curr).addScaledVector(side, -roadWidth / 2);
      const rightEdge = new THREE.Vector3().copy(curr).addScaledVector(side, roadWidth / 2);

      roadPositions.push(leftEdge.x, leftEdge.y, leftEdge.z);
      roadPositions.push(rightEdge.x, rightEdge.y, rightEdge.z);

      // Deep graphite asphalt with high-end matte finish
      const baseR = 0.12;
      const baseG = 0.12;
      const baseB = 0.14;
      roadColors.push(baseR, baseG, baseB);
      roadColors.push(baseR, baseG, baseB);

      // TrackSims Colored Dominance Center Ribbon (width: 14 units, elevated +0.8)
      const domLeft = new THREE.Vector3().copy(curr).addScaledVector(side, -6).add(new THREE.Vector3(0, 0.8, 0));
      const domRight = new THREE.Vector3().copy(curr).addScaledVector(side, 6).add(new THREE.Vector3(0, 0.8, 0));

      dominanceOverlayPositions.push(domLeft.x, domLeft.y, domLeft.z);
      dominanceOverlayPositions.push(domRight.x, domRight.y, domRight.z);

      dominanceOverlayColors.push(domColor.r, domColor.g, domColor.b);
      dominanceOverlayColors.push(domColor.r, domColor.g, domColor.b);

      // Outer Kerb boundary lines
      const kLeft = new THREE.Vector3().copy(leftEdge).addScaledVector(side, -2).add(new THREE.Vector3(0, 0.4, 0));
      const kRight = new THREE.Vector3().copy(rightEdge).addScaledVector(side, 2).add(new THREE.Vector3(0, 0.4, 0));
      kerbLeftPositions.push(kLeft.x, kLeft.y, kLeft.z);
      kerbRightPositions.push(kRight.x, kRight.y, kRight.z);
    }

    // Indices for ribbon meshes
    const roadIndices: number[] = [];
    for (let i = 0; i < smoothPoints.length; i++) {
      const p1 = i * 2;
      const p2 = i * 2 + 1;
      const p3 = ((i + 1) % smoothPoints.length) * 2;
      const p4 = ((i + 1) % smoothPoints.length) * 2 + 1;

      roadIndices.push(p1, p2, p3);
      roadIndices.push(p2, p4, p3);
    }

    // 8a. Asphalt Base Mesh
    const roadGeo = new THREE.BufferGeometry();
    roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(roadPositions, 3));
    roadGeo.setAttribute("color", new THREE.Float32BufferAttribute(roadColors, 3));
    roadGeo.setIndex(roadIndices);
    roadGeo.computeVertexNormals();

    const roadMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.88,
      metalness: 0.15,
      side: THREE.DoubleSide,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    scene.add(roadMesh);

    // 8b. Extruded Velocity Dominance Ribbon
    const domGeo = new THREE.BufferGeometry();
    domGeo.setAttribute("position", new THREE.Float32BufferAttribute(dominanceOverlayPositions, 3));
    domGeo.setAttribute("color", new THREE.Float32BufferAttribute(dominanceOverlayColors, 3));
    domGeo.setIndex(roadIndices);
    domGeo.computeVertexNormals();

    const domMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.35,
      metalness: 0.6,
      emissive: new THREE.Color(0x222222),
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.92,
    });
    const domMesh = new THREE.Mesh(domGeo, domMat);
    scene.add(domMesh);

    // 8c. Crisp Titanium Kerb Edges
    const kerbMat = new THREE.LineBasicMaterial({
      color: 0x8e8e93,
      linewidth: 1.5,
      transparent: true,
      opacity: 0.7,
    });
    const leftKerbGeo = new THREE.BufferGeometry().setFromPoints(
      smoothPoints.map((_, i) => new THREE.Vector3(kerbLeftPositions[i * 3], kerbLeftPositions[i * 3 + 1], kerbLeftPositions[i * 3 + 2]))
    );
    const rightKerbGeo = new THREE.BufferGeometry().setFromPoints(
      smoothPoints.map((_, i) => new THREE.Vector3(kerbRightPositions[i * 3], kerbRightPositions[i * 3 + 1], kerbRightPositions[i * 3 + 2]))
    );
    scene.add(new THREE.LineLoop(leftKerbGeo, kerbMat));
    scene.add(new THREE.LineLoop(rightKerbGeo, kerbMat));

    // 9. Start / Finish Gate Marker
    const startPt = smoothPoints[0];
    const startNext = smoothPoints[1];
    const startTan = new THREE.Vector3().subVectors(startNext, startPt).normalize();
    const startSide = new THREE.Vector3().crossVectors(startTan, new THREE.Vector3(0, 1, 0)).normalize();

    const finishLineGeo = new THREE.BoxGeometry(roadWidth + 4, 2.5, 4);
    const finishLineMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.8,
    });
    const finishGate = new THREE.Mesh(finishLineGeo, finishLineMat);
    finishGate.position.copy(startPt);
    finishGate.position.y += 1.2;
    finishGate.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), startSide);
    scene.add(finishGate);

    // 10. Car Markers for Each Driver (up to 4)
    const carMarkers: THREE.Group[] = [];
    const lateralOffsets = [-9, 9, -15, 15]; // Stagger cars laterally across track width

    drivers.forEach((drv, idx) => {
      const group = new THREE.Group();
      const col = new THREE.Color(drv.color || "#ffffff");

      // Core Glowing Orb
      const sphereGeo = new THREE.SphereGeometry(18, 24, 24);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.9,
        roughness: 0.1,
        metalness: 0.8,
      });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      group.add(sphere);

      // Outer Halo Ring
      const ringGeo = new THREE.RingGeometry(24, 29, 28);
      const ringMat = new THREE.MeshBasicMaterial({
        color: col,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.75,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      // Drop Stalk / Light Shaft
      const stalkGeo = new THREE.CylinderGeometry(1.2, 1.2, 20, 8);
      const stalkMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.6,
      });
      const stalk = new THREE.Mesh(stalkGeo, stalkMat);
      stalk.position.y = -10;
      group.add(stalk);

      group.position.copy(startPt);
      group.position.y += 18;
      scene.add(group);
      carMarkers.push(group);
    });

    // 11. Orbit Camera Controls State
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

    // 12. High-Performance Animation Loop
    let animId: number;
    let lastTime = performance.now();

    const animate = (time: number) => {
      animId = requestAnimationFrame(animate);
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      // Auto-Orbit cinematic spin if enabled
      if (stateRef.current.autoRotate && !controlsState.isDragging) {
        controlsState.theta += 0.15 * dt;
        updateCameraPos();
      }

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

      // Update position of all car markers smoothly
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
            group.position.lerp(targetPos, 0.4);
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
      domGeo.dispose();
      domMat.dispose();
      leftKerbGeo.dispose();
      rightKerbGeo.dispose();
      kerbMat.dispose();
      groundGeo.dispose();
      groundMat.dispose();
      groundTex.dispose();
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

  // Active sample telemetry at current distance
  const currentSample = useMemo(() => {
    if (!sceneRef.current || sceneRef.current.pointsMap.length === 0) return null;
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
    return pts[closestIdx] || null;
  }, [currentDist]);

  // Sector identification (Sakhir standard: S1: 0-1920m, S2: 1920-3790m, S3: 3790-5412m)
  const currentSector = useMemo(() => {
    if (currentDist < 1920) return "S1";
    if (currentDist < 3790) return "S2";
    return "S3";
  }, [currentDist]);

  // Speed Delta Calculation between Driver 1 and Driver 2
  const speedDelta = useMemo(() => {
    if (!currentSample || currentSample.speeds.length < 2) return null;
    const spd1 = currentSample.speeds[0];
    const spd2 = currentSample.speeds[1];
    const diff = spd1 - spd2;
    return {
      diff: Math.round(diff),
      leader: diff >= 0 ? drivers[0]?.abbreviation : drivers[1]?.abbreviation,
      leaderColor: diff >= 0 ? drivers[0]?.color : drivers[1]?.color,
    };
  }, [currentSample, drivers]);

  return (
    <div className="relative rounded-2xl border border-white/[0.08] bg-[#070709] overflow-hidden shadow-2xl">
      {/* 1. TrackSims Floating Top Comparison HUD Bar */}
      <div className="absolute top-4 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Track Title & Circuit Sector Badges */}
        <div className="flex items-center gap-2.5 pointer-events-auto bg-[#101014]/90 backdrop-blur-xl px-3.5 py-1.5 rounded-xl border border-white/[0.09] shadow-xl">
          <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-neutral-200">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Bahrain International Circuit (Sakhir)
          </span>
          <span className="text-white/20">|</span>
          {/* Mini Sector Badges */}
          <div className="flex items-center gap-1">
            {(["S1", "S2", "S3"] as const).map((sec) => (
              <span
                key={sec}
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-all ${
                  currentSector === sec
                    ? "bg-amber-400/20 text-amber-300 border border-amber-400/40"
                    : "bg-white/[0.04] text-neutral-500 border border-white/[0.05]"
                }`}
              >
                {sec}
              </span>
            ))}
          </div>
        </div>

        {/* Dynamic Speed Delta Pill & Driver Comparison Bar */}
        {speedDelta && currentSample && (
          <div className="flex items-center gap-2 pointer-events-auto bg-[#101014]/90 backdrop-blur-xl px-4 py-1.5 rounded-xl border border-white/[0.09] shadow-xl">
            {/* Driver 1 Speed */}
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: drivers[0]?.color }} />
              <span className="text-xs font-bold" style={{ color: drivers[0]?.color }}>
                {drivers[0]?.abbreviation}
              </span>
              <span className="text-xs font-bold text-white">
                {Math.round(currentSample.speeds[0] || 0)} <span className="text-[10px] text-neutral-400 font-normal">km/h</span>
              </span>
            </div>

            {/* Speed Delta Pill */}
            <div
              className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-extrabold border"
              style={{
                backgroundColor: `${speedDelta.leaderColor}20`,
                borderColor: `${speedDelta.leaderColor}50`,
                color: speedDelta.leaderColor,
              }}
            >
              {speedDelta.leader} {speedDelta.diff >= 0 ? `+${speedDelta.diff}` : speedDelta.diff} km/h
            </div>

            {/* Driver 2 Speed */}
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: drivers[1]?.color }} />
              <span className="text-xs font-bold" style={{ color: drivers[1]?.color }}>
                {drivers[1]?.abbreviation}
              </span>
              <span className="text-xs font-bold text-white">
                {Math.round(currentSample.speeds[1] || 0)} <span className="text-[10px] text-neutral-400 font-normal">km/h</span>
              </span>
            </div>

            {/* Driver 3 / Driver 4 if present */}
            {drivers.slice(2).map((d, i) => (
              <div key={d.abbreviation} className="hidden lg:flex items-center gap-2 font-mono border-l border-white/10 pl-2">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-xs font-bold" style={{ color: d.color }}>
                  {d.abbreviation}
                </span>
                <span className="text-xs font-bold text-white">
                  {Math.round(currentSample.speeds[i + 2] || 0)} <span className="text-[10px] text-neutral-400 font-normal">km/h</span>
                </span>
              </div>
            ))}
          </div>
        )}

        {/* View & Camera Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-1.5 rounded-xl text-xs font-medium border transition-all shadow-md ${
              autoRotate
                ? "bg-blue-500/20 text-blue-400 border-blue-500/40"
                : "bg-[#101014]/90 text-neutral-400 hover:text-white border-white/[0.08]"
            }`}
            title="Toggle Cinematic Auto-Orbit"
          >
            <Sparkles className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleResetCamera}
            className="p-1.5 rounded-xl text-xs font-medium bg-[#101014]/90 hover:bg-[#1a1a20] text-neutral-400 hover:text-white border border-white/[0.08] transition-all shadow-md"
            title="Reset Camera View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleViewToggle}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-[#101014]/90 hover:bg-[#1a1a20] text-neutral-300 hover:text-white border border-white/[0.08] transition-all shadow-md"
          >
            <Compass className="w-3.5 h-3.5" />
            {viewMode} View
          </button>
        </div>
      </div>

      {/* 2. TrackSims Live RUNNING GAP Overlay (Bottom-Left) */}
      <div className="absolute bottom-20 left-4 z-10 pointer-events-none flex flex-col gap-1.5">
        <div className="bg-[#101014]/90 backdrop-blur-xl px-4 py-2 rounded-xl border border-white/[0.09] shadow-2xl flex items-center gap-3">
          <div>
            <div className="text-[9px] font-mono uppercase tracking-widest text-neutral-400 font-bold">
              Running Gap
            </div>
            <div className="text-base font-mono font-black text-white">
              {currentSample?.deltaSeconds !== undefined
                ? `${currentSample.deltaSeconds >= 0 ? "+" : ""}${currentSample.deltaSeconds.toFixed(3)}s`
                : "+0.000s"}
            </div>
          </div>
          <div className="h-7 w-px bg-white/10" />
          <div>
            <div className="text-[9px] font-mono uppercase tracking-widest text-neutral-400 font-bold">
              Dominance
            </div>
            <div className="text-xs font-mono font-bold flex items-center gap-1" style={{ color: drivers[currentSample?.dominantDriverIdx || 0]?.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: drivers[currentSample?.dominantDriverIdx || 0]?.color }} />
              {drivers[currentSample?.dominantDriverIdx || 0]?.abbreviation || "VER"}
            </div>
          </div>
        </div>
      </div>

      {/* 3. 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full h-[500px] cursor-grab active:cursor-grabbing" />

      {/* 4. Bottom Scrubber & Playback Controls */}
      <div className="p-4 bg-[#09090b]/95 border-t border-white/[0.06] flex flex-col md:flex-row items-center gap-4">
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-9 h-9 rounded-full bg-white text-black hover:bg-neutral-200 flex items-center justify-center transition-transform active:scale-95 shadow-lg"
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
            title="Reset to Start/Finish Line"
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

        {/* Scrub Slider with Distance Markers */}
        <div className="flex-1 w-full flex items-center gap-3">
          <span className="text-xs font-mono text-neutral-400 w-16 text-right shrink-0">
            {Math.round(currentDist).toLocaleString()}m
          </span>
          <div className="relative flex-1 flex items-center">
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
              className="w-full accent-red-500 h-1.5 bg-white/10 rounded-lg cursor-pointer transition-all"
            />
          </div>
          <span className="text-xs font-mono text-neutral-500 w-16 shrink-0">
            {Math.round(maxD).toLocaleString()}m
          </span>
        </div>
      </div>
    </div>
  );
}

