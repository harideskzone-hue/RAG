import { useState } from 'react';
import { ChatResponse, Evidence } from '../../api/client';
import { Camera, Clock, Video, User, X, ShieldCheck, AlertTriangle, Play } from 'lucide-react';

interface EvidencePanelProps {
  selectedContract: ChatResponse | null;
}

export function EvidencePanel({ selectedContract }: EvidencePanelProps) {
  const [activeVideoModal, setActiveVideoModal] = useState<string | null>(null);

  if (!selectedContract) {
    return (
      <div className="evidence-panel empty">
        <div className="empty-state">
          <Camera size={40} className="empty-icon" />
          <p>Select any investigation response to inspect verified visual evidence</p>
        </div>
      </div>
    );
  }

  const { evidence, timeline, grounding_status, detection_status, person_count, zone, evaluation_window, scene_clip, scene_thumbnail } = selectedContract;
  const status = detection_status || (evidence && evidence.length > 0 ? 'DETECTED' : 'EMPTY');
  const count = person_count ?? evidence?.length ?? 0;
  const zoneName = zone || 'Entrance (cam_auto_01)';
  const windowStr = evaluation_window || '00:00 - 01:50';
  const mainClipUrl = scene_clip || (evidence && evidence.length > 0 ? evidence[0].clip_url : '/media/videos/completed/VIDEO-2026-08-11-12-15-36.mp4');

  return (
    <div className="evidence-panel">
      <div className="evidence-header">
        <div className="evidence-header-left">
          <h2>AUTHORITATIVE EVIDENCE</h2>
          <span className="zone-tag">{zoneName}</span>
        </div>
        {grounding_status === 'VALID' && <span className="status-badge valid">GROUNDED</span>}
        {grounding_status === 'INVALID' && <span className="status-badge invalid">UNGROUNDED</span>}
        {grounding_status === 'ABSTAIN' && <span className="status-badge abstain">ABSTAINED</span>}
      </div>

      <div className="evidence-content-scroll">
        {/* Primary Entrance CCTV Video Evidence Box */}
        {mainClipUrl && (
          <div className="scene-video-card">
            <div className="scene-video-header">
              <div className="scene-video-info">
                <span className="scene-video-title">🎥 Entrance CCTV Video Evidence</span>
                <span className="scene-video-window">Evaluation Window: {windowStr}</span>
              </div>
              <button
                type="button"
                className="scene-play-btn"
                onClick={() => setActiveVideoModal(mainClipUrl)}
              >
                <Play size={13} fill="currentColor" /> Play Footage
              </button>
            </div>
          </div>
        )}

        {/* State-Driven Evidence Content */}
        {status === 'DETECTED' && evidence && evidence.length > 0 ? (
          <div className="evidence-section">
            <div className="section-title-row">
              <h3>Verified Person Gallery ({evidence.length})</h3>
              <span className="section-sub-badge">{count} {count === 1 ? 'Person' : 'Persons'} Identified</span>
            </div>

            <div className="evidence-list">
              {evidence.map((ev: Evidence, idx: number) => {
                const cropUrl = ev.crop_url ? ev.crop_url : null;
                const clipUrl = ev.clip_url ? ev.clip_url : mainClipUrl;

                return (
                  <div key={idx} className="evidence-card">
                    <div className="crop-container">
                      {cropUrl ? (
                        <img
                          src={cropUrl}
                          alt={ev.person_id || `Evidence ${idx + 1}`}
                          className="evidence-crop-img"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                            const parent = (e.target as HTMLElement).parentElement;
                            if (parent) {
                              parent.innerHTML = '<div class="crop-fallback"><span class="crop-text">CROP</span></div>';
                            }
                          }}
                        />
                      ) : (
                        <div className="crop-fallback">
                          <User size={20} className="crop-icon" />
                          <span className="crop-text">{ev.person_id ? ev.person_id.slice(0, 10) : 'CROP'}</span>
                        </div>
                      )}
                    </div>

                    <div className="evidence-details">
                      {ev.person_id && (
                        <div className="detail-row person-id-row">
                          <User size={12} className="text-blue-400" />
                          <strong className="person-id-label">{ev.person_id}</strong>
                        </div>
                      )}
                      <div className="detail-row">
                        <Camera size={12}/> <span>{ev.camera_id || 'cam_auto_01'}</span>
                      </div>
                      <div className="detail-row">
                        <Clock size={12}/> <span className="font-mono text-xs">{ev.timestamp || '00:00'}</span>
                      </div>
                      <div className="detail-row">
                        <span className="source-label">Source:</span> <span>{ev.source || 'Video Analysis'}</span>
                      </div>

                      {clipUrl && (
                        <button
                          type="button"
                          className="video-link-btn"
                          onClick={() => setActiveVideoModal(clipUrl)}
                        >
                          <Video size={12} /> Play Evidence Clip
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : status === 'EMPTY' ? (
          <div className="empty-scene-verification">
            <div className="empty-scene-header">
              <ShieldCheck size={28} className="text-zinc-400" />
              <h4>Entrance Scene Clear</h4>
              <p>Authoritative CV pipeline verified 0 persons in frame during evaluation window <strong>{windowStr}</strong>.</p>
            </div>
            {scene_thumbnail && (
              <div className="empty-snapshot-container">
                <span className="snapshot-label">Scene Background Keyframe:</span>
                <img src={scene_thumbnail} alt="Empty entrance scene" className="empty-scene-img" />
              </div>
            )}
          </div>
        ) : status === 'ABSTAINED' ? (
          <div className="abstained-scene-card">
            <AlertTriangle size={24} className="text-amber-400" />
            <h4>System Abstention</h4>
            <p>The system abstained from making a presence or absence assertion because available footage does not satisfy grounding confidence thresholds.</p>
          </div>
        ) : (
          <p className="no-data">No observational evidence for this query.</p>
        )}

        {timeline && timeline.length > 0 && (
          <div className="evidence-section">
            <h3>Observation Timeline</h3>
            <ul className="timeline-list">
              {timeline.map((event: any, idx: number) => (
                <li key={idx} className="timeline-item">
                  <span className="time">{event.timestamp}</span>
                  <span className="desc">{event.description || event.event}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Interactive Video Evidence Modal */}
      {activeVideoModal && (
        <div className="video-modal-overlay" onClick={() => setActiveVideoModal(null)}>
          <div className="video-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="video-modal-header">
              <div className="modal-title-group">
                <Video size={16} className="text-blue-400" />
                <span className="font-semibold text-sm">Authoritative CCTV Video Evidence</span>
              </div>
              <button type="button" className="close-btn" onClick={() => setActiveVideoModal(null)}>
                <X size={16} />
              </button>
            </div>
            <div className="video-player-wrapper">
              <video controls autoPlay className="evidence-video-player" src={activeVideoModal}>
                Your browser does not support playing this video.
              </video>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
