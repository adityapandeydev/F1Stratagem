import axios from "axios";
import type {
  Season, Event, Session, SessionResult, Lap,
  TelemetryComparison, WeatherData, IngestionJob,
  Driver, Team,
} from "../types";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8080/api/v1",
  headers: { "Content-Type": "application/json" },
});

export const api = {
  // Seasons
  getSeasons: () => client.get<Season[]>("/seasons").then((r) => r.data),
  syncSeasonSchedule: (year: number) =>
    client.post(`/seasons/${year}/sync`).then((r) => r.data),
  getSeasonByYear: (year: number) =>
    client.get<Season>(`/seasons/${year}`).then((r) => r.data),

  // Events
  getEventsBySeason: (year: number) =>
    client.get<Event[]>(`/seasons/${year}/events`).then((r) => r.data),

  // Sessions
  getSessionsByEvent: (eventId: number) =>
    client.get<Session[]>(`/events/${eventId}/sessions`).then((r) => r.data),
  getSession: (sessionId: number) =>
    client.get<Session>(`/sessions/${sessionId}`).then((r) => r.data),

  // Session Data
  getSessionResults: (sessionId: number) =>
    client.get<SessionResult[]>(`/sessions/${sessionId}/results`).then((r) => r.data),
  getSessionLaps: (sessionId: number, driverId?: number) =>
    client.get<Lap[]>(`/sessions/${sessionId}/laps`, {
      params: driverId ? { driver_id: driverId } : undefined,
    }).then((r) => r.data),
  getSessionWeather: (sessionId: number) =>
    client.get<WeatherData[]>(`/sessions/${sessionId}/weather`).then((r) => r.data),
  getSessionStints: (sessionId: number) =>
    client.get<any[]>(`/sessions/${sessionId}/stints`).then((r) => r.data),

  // Telemetry
  getTelemetryComparison: (
    sessionId: number,
    driver1: string,
    driver2: string,
    lap1?: number,
    lap2?: number
  ) =>
    client
      .get<TelemetryComparison>(`/sessions/${sessionId}/telemetry`, {
        params: { driver1, driver2, lap1, lap2 },
      })
      .then((r) => r.data),

  // Ingestion
  triggerIngestion: (sessionId: number) =>
    client.post<IngestionJob>(`/sessions/${sessionId}/ingest`).then((r) => r.data),
  getIngestionStatus: (sessionId: number) =>
    client.get<IngestionJob>(`/sessions/${sessionId}/ingest/status`).then((r) => r.data),

  // Drivers / Teams
  getDrivers: () => client.get<Driver[]>("/drivers").then((r) => r.data),
  getTeams: () => client.get<Team[]>("/teams").then((r) => r.data),
};
