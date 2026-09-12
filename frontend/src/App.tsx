import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/layout/Layout";
import HomePage from "./pages/HomePage";
import SessionPage from "./pages/SessionPage";
import ComparePage from "./pages/ComparePage";
import ErrorBoundary from "./components/ui/ErrorBoundary";

function ComingSoonPage({ title, phase }: { title: string; phase: string }) {
  return (
    <div className="p-12 text-center rounded-3xl border border-white/[0.08] bg-[#0a0a0b]/80 backdrop-blur-xl max-w-lg mx-auto space-y-3 my-12 shadow-2xl">
      <div className="inline-block px-3 py-1 rounded-full text-[11px] font-mono font-bold bg-white/10 text-white uppercase tracking-wider">
        {phase}
      </div>
      <h2 className="text-2xl font-bold text-white">{title}</h2>
      <p className="text-sm text-neutral-400">
        This section is scheduled for {phase} and will be available once the corresponding data models and analyses are connected.
      </p>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/season/:year" element={<HomePage />} />
          
          {/* Telemetry Comparison Routes */}
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/compare/:sessionId" element={<ComparePage />} />
          <Route path="/telemetry" element={<Navigate to="/compare" replace />} />

          {/* Session Routes */}
          <Route path="/session" element={<SessionPage />} />
          <Route path="/session/:id" element={<SessionPage />} />
          <Route path="/session/:sessionId" element={<SessionPage />} />

          {/* Later Phase Routes */}
          <Route path="/circuits" element={<ComingSoonPage title="Circuit Geometries & Track Maps" phase="Phase 2" />} />
          <Route path="/drivers" element={<ComingSoonPage title="Driver Profiles & Head-to-Head Index" phase="Phase 2" />} />
          <Route path="/teams" element={<ComingSoonPage title="Constructor Engineering & Specs" phase="Phase 2" />} />
          <Route path="/standings" element={<ComingSoonPage title="Championship Standings" phase="Phase 2" />} />

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}
