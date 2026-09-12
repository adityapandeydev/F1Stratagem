package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Minute, // FastF1 session load can take time
		},
	}
}

type ExtractSessionPayload struct {
	Year        int    `json:"year"`
	RoundNumber int    `json:"round_number"`
	Session     string `json:"session"`
}

type CompareTelemetryPayload struct {
	Year        int    `json:"year"`
	RoundNumber int    `json:"round_number"`
	Session     string `json:"session"`
	Driver1     string `json:"driver1"`
	Lap1        int    `json:"lap1"`
	Driver2     string `json:"driver2"`
	Lap2        int    `json:"lap2"`
}

type WorkerResponse struct {
	Status string                 `json:"status"`
	Data   map[string]interface{} `json:"data"`
}

func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/health", c.baseURL), nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("worker health check returned %d", resp.StatusCode)
	}
	return nil
}

func (c *Client) GetSchedule(ctx context.Context, year int) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/schedule/%d", c.baseURL, year), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode schedule response: %w", err)
	}
	return result, nil
}

func (c *Client) ExtractSession(ctx context.Context, year int, round int, sessionType string) (*WorkerResponse, error) {
	payload := ExtractSessionPayload{
		Year:        year,
		RoundNumber: round,
		Session:     sessionType,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/session/extract", c.baseURL), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("worker returned status %d", resp.StatusCode)
	}

	var res WorkerResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("decode extract response: %w", err)
	}
	return &res, nil
}

func (c *Client) CompareTelemetry(ctx context.Context, year int, round int, sessionType string, d1 string, lap1 int, d2 string, lap2 int) (map[string]interface{}, error) {
	payload := CompareTelemetryPayload{
		Year:        year,
		RoundNumber: round,
		Session:     sessionType,
		Driver1:     d1,
		Lap1:        lap1,
		Driver2:     d2,
		Lap2:        lap2,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/telemetry/compare", c.baseURL), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("worker compare returned status %d", resp.StatusCode)
	}

	var res map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("decode compare response: %w", err)
	}
	return res, nil
}
