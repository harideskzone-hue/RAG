export interface Citation {
  source: string;
  content: string;
  confidence: number;
}

export interface Evidence {
  evidence_id: string;
  source: string;
  camera_id: string | null;
  timestamp: string | null;
  description: string | null;
  confidence: number;
  crop_url?: string | null;
  clip_url?: string | null;
  person_id?: string | null;
  track_id?: string | null;
}

export interface ExecutionStep {
  name: string;
  status: string;
  latency_ms: number;
  error: string | null;
}

export interface ChatResponse {
  status: string;
  detection_status: 'DETECTED' | 'EMPTY' | 'ABSTAINED' | 'ERROR' | 'CRITICAL_ALERT' | 'INCIDENT_ALERT';
  person_count: number;
  zone: string;
  evaluation_window?: string;
  scene_clip?: string;
  scene_thumbnail?: string;
  thought?: string | null;
  thinking_process?: string | null;
  answer: string | null;
  grounding_status: string;
  confidence: number;
  citations: Citation[];
  evidence: Evidence[];
  timeline: any[];
  processing: any;
  execution: {
    status: string;
    steps: ExecutionStep[];
  };
  processing_time_ms: number;
  trace_id: string;
}

export class VistaClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  async chat(query: string, videoId?: string): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer dev_token',
      },
      body: JSON.stringify({
        query,
        video_id: videoId,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }
}

export const apiClient = new VistaClient();
