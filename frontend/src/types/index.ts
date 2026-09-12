export interface Season {
  id: number;
  year: number;
}

export interface Circuit {
  id: number;
  circuit_key: string;
  name: string;
  country: string;
  location?: string;
  latitude?: number;
  longitude?: number;
  track_length_m?: number;
  num_corners?: number;
  num_drs_zones?: number;
}

export interface Event {
  id: number;
  season_id: number;
  circuit_id?: number;
  round_number: number;
  event_name: string;
  official_name?: string;
  country?: string;
  location?: string;
  event_format: string;
  event_date?: string;
  circuit?: Circuit;
}

export interface Session {
  id: number;
  event_id: number;
  session_name: string;
  session_type: string;
  session_date?: string;
  total_laps?: number;
  data_status: string;
  data_error?: string;
  event?: Event;
}

export interface Driver {
  id: number;
  driver_number?: string;
  abbreviation: string;
  first_name: string;
  last_name: string;
  full_name?: string;
  country_code?: string;
  headshot_url?: string;
}

export interface Team {
  id: number;
  name: string;
  color?: string;
  country?: string;
}

export interface SessionResult {
  id: number;
  session_id: number;
  driver_id: number;
  position?: number;
  grid_position?: number;
  q1_time_ms?: number;
  q2_time_ms?: number;
  q3_time_ms?: number;
  race_time_ms?: number;
  status?: string;
  points: number;
  driver?: Driver;
  team?: Team;
}

export interface Lap {
  id: number;
  session_id: number;
  driver_id: number;
  lap_number: number;
  lap_time_ms?: number;
  sector1_ms?: number;
  sector2_ms?: number;
  sector3_ms?: number;
  compound?: string;
  tyre_life?: number;
  stint?: number;
  position?: number;
  is_personal_best: boolean;
  is_deleted: boolean;
  speed_st?: number;
}

export interface TelemetryPoint {
  d: number;    // distance in meters
  spd: number;  // speed in km/h
  thr: number;  // throttle 0-100%
  brk: number;  // brake 0/1
  rpm: number;  // RPM
  gear: number; // gear 1-8
  drs: number;  // DRS status
  x: number;    // X coordinate on track
  y: number;    // Y coordinate on track
  z: number;    // Z elevation
}

export interface TelemetryDriverMeta {
  abbreviation: string;
  full_name: string;
  team_name: string;
  color: string;
  lap_number: number;
  lap_time_ms: number;
}

export interface MultiTelemetryComparison {
  drivers: TelemetryDriverMeta[];
  telemetry: TelemetryPoint[][];
  time_delta: number[];
}

export interface TelemetryComparison {
  // Legacy
  driver1?: Driver | TelemetryDriverMeta;
  driver2?: Driver | TelemetryDriverMeta;
  lap1?: number;
  lap2?: number;
  telemetry1?: TelemetryPoint[];
  telemetry2?: TelemetryPoint[];

  // Multi-driver (up to 4)
  drivers?: TelemetryDriverMeta[];
  telemetry?: TelemetryPoint[][];
  time_delta?: number[];
}

export interface WeatherData {
  id?: number;
  session_id?: number;
  time_offset_ms: number;
  air_temp?: number;
  track_temp?: number;
  humidity?: number;
  pressure?: number;
  wind_speed?: number;
  wind_direction?: number;
  rainfall: boolean;
}

export interface IngestionJob {
  id: string;
  session_id: number;
  status: string;
  progress: number;
  error_message?: string;
}
