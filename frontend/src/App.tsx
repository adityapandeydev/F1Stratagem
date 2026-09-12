import { Routes, Route } from "react-router-dom";
import Layout from "./components/layout/Layout";
import HomePage from "./pages/HomePage";
import SessionPage from "./pages/SessionPage";
import ComparePage from "./pages/ComparePage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/season/:year" element={<HomePage />} />
        <Route path="/session/:sessionId" element={<SessionPage />} />
        <Route path="/compare/:sessionId" element={<ComparePage />} />
      </Route>
    </Routes>
  );
}