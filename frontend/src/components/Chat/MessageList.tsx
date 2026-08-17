import { useState, useEffect } from 'react';
import { 
  Check, 
  Database, 
  Calendar, 
  AlertTriangle,
  ChevronDown, 
  ChevronRight, 
  Clock, 
  UserCheck,
  ShieldCheck,
  ShieldAlert,
  Loader2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatResponse } from '../../api/client';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  responseContract?: ChatResponse;
}

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  onSelectEvidence: (contract: ChatResponse) => void;
}

const THINKING_STAGES = [
  'Analyzing query intent & spatial constraints...',
  'Querying OSNet MSMT17 embeddings in vector store...',
  'Running deterministic video provenance validation gates...',
  'Fusing tracklet identities & resolving canonical persons...',
  'Reasoning over verified CCTV observations...',
  'Synthesizing verified intelligence response...'
];

export function MessageList({ messages, isLoading, onSelectEvidence }: MessageListProps) {
  // Professional Claude Agent Thinking Component
  const ClaudeAgentThinking = () => {
    const [elapsed, setElapsed] = useState(0.0);
    const [stageIndex, setStageIndex] = useState(0);

    useEffect(() => {
      const timer = setInterval(() => {
        setElapsed((prev) => +(prev + 0.1).toFixed(1));
      }, 100);

      const stageTimer = setInterval(() => {
        setStageIndex((prev) => (prev + 1) % THINKING_STAGES.length);
      }, 1500);

      return () => {
        clearInterval(timer);
        clearInterval(stageTimer);
      };
    }, []);

    return (
      <div className="claude-agent-thinking">
        <div className="thinking-main-row">
          <div className="thinking-indicator-ring">
            <Loader2 size={16} className="thinking-spinner" />
          </div>
          <div className="thinking-text-group">
            <span className="thinking-title">Thinking</span>
            <span className="thinking-current-stage">{THINKING_STAGES[stageIndex]}</span>
          </div>
          <div className="thinking-timer">
            <Clock size={12} />
            <span>{elapsed.toFixed(1)}s</span>
          </div>
        </div>
      </div>
    );
  };

  // Clean Completed Thought Accordion (100% LLM Thinking & Validation Trace)
  const ClaudeCompletedThought = ({ contract }: { contract: ChatResponse }) => {
    const [isOpen, setIsOpen] = useState(false);
    const totalTimeSec = (contract.processing_time_ms ? contract.processing_time_ms / 1000 : 0.8).toFixed(2);
    const steps = contract.execution?.steps || [];
    const thoughtText = contract.thought || contract.thinking_process;

    return (
      <div className="thought-accordion">
        <button
          type="button"
          className="thought-accordion-trigger"
          onClick={() => setIsOpen(!isOpen)}
        >
          <div className="thought-trigger-left">
            <span className="thought-bullet">•</span>
            <span className="thought-label">
              Thought for <strong>{totalTimeSec}s</strong> ({steps.length} validation stages)
            </span>
          </div>
          <div className="thought-trigger-right">
            {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>
        </button>

        {isOpen && (
          <div className="thought-accordion-body">
            {thoughtText && (
              <div className="thought-reasoning-block">
                <div className="thought-reasoning-title">Chain of Thought Reasoning:</div>
                <div className="thought-reasoning-text">
                  {thoughtText.split('\n').map((line, i) => (
                    <p key={i} className="thought-reasoning-line">{line}</p>
                  ))}
                </div>
              </div>
            )}

            <div className="thought-stages-title">Agent Execution Ledger:</div>
            <div className="thought-stages-list">
              {steps.map((step, idx) => (
                <div key={idx} className="thought-stage-row">
                  <div className="thought-stage-icon">
                    {step.status === 'completed' ? (
                      <Check size={12} className="text-emerald-400" />
                    ) : (
                      <span className="stage-dot" />
                    )}
                  </div>
                  <span className="stage-name">{step.name}</span>
                  {step.latency_ms > 0 && (
                    <span className="stage-time">{step.latency_ms}ms</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Explicit Contract Detection Status Banner
  const DetectionStatusBanner = ({ contract }: { contract: ChatResponse }) => {
    const status = contract.detection_status || (contract.evidence?.length > 0 ? 'DETECTED' : 'EMPTY');
    const personCount = contract.person_count ?? contract.evidence?.length ?? 0;
    const zone = contract.zone || 'Entrance (cam_auto_01)';
    const window = contract.evaluation_window || '00:00 - 01:50';

    if (status === 'CRITICAL_ALERT' || status === 'INCIDENT_ALERT') {
      return (
        <div className="detection-banner banner-critical-alert" style={{
          background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.08) 100%)',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          boxShadow: '0 0 15px rgba(239, 68, 68, 0.15)'
        }}>
          <div className="banner-header">
            <div className="banner-title-wrap">
              <ShieldAlert size={18} className="text-red-400 animate-pulse" />
              <span className="banner-title text-red-400 font-bold tracking-wide">🚨 CRITICAL INCIDENT ALERT: ROBBERY / CHAIN SNATCHING DETECTED</span>
            </div>
            <div className="banner-count-badge" style={{ background: 'rgba(239, 68, 68, 0.25)', color: '#fca5a5', borderColor: 'rgba(239, 68, 68, 0.5)' }}>
              <strong>10s Incident Clip</strong>
            </div>
          </div>
          <div className="banner-meta-row" style={{ color: '#fecaca' }}>
            <span><strong>Zone:</strong> {zone}</span>
            <span><strong>Window:</strong> {window}</span>
          </div>
        </div>
      );
    }

    if (status === 'DETECTED') {
      return (
        <div className="detection-banner banner-detected">
          <div className="banner-header">
            <div className="banner-title-wrap">
              <UserCheck size={16} className="text-emerald-400" />
              <span className="banner-title">PERSON(S) DETECTED</span>
            </div>
            <div className="banner-count-badge">
              <strong>{personCount}</strong> {personCount === 1 ? 'Person' : 'Persons'}
            </div>
          </div>
          <div className="banner-meta-row">
            <span><strong>Zone:</strong> {zone}</span>
            <span><strong>Window:</strong> {window}</span>
          </div>
        </div>
      );
    }

    if (status === 'EMPTY') {
      return (
        <div className="detection-banner banner-empty">
          <div className="banner-header">
            <div className="banner-title-wrap">
              <ShieldCheck size={16} className="text-zinc-400" />
              <span className="banner-title">ENTRANCE CLEAR / NO PERSONS PRESENT</span>
            </div>
            <div className="banner-count-badge badge-neutral">
              0 Persons
            </div>
          </div>
          <div className="banner-meta-row">
            <span><strong>Zone:</strong> {zone}</span>
            <span><strong>Evaluated Window:</strong> {window}</span>
          </div>
        </div>
      );
    }

    if (status === 'ABSTAINED') {
      return (
        <div className="detection-banner banner-abstained">
          <div className="banner-header">
            <div className="banner-title-wrap">
              <AlertTriangle size={16} className="text-amber-400" />
              <span className="banner-title">EVIDENCE INSUFFICIENT / SYSTEM ABSTAINED</span>
            </div>
          </div>
          <div className="banner-meta-row">
            <span>Integrity protection active: Insufficient authoritative footage to confirm presence or absence.</span>
          </div>
        </div>
      );
    }

    return (
      <div className="detection-banner banner-error">
        <div className="banner-header">
          <div className="banner-title-wrap">
            <ShieldAlert size={16} className="text-rose-400" />
            <span className="banner-title">VERIFICATION ERROR</span>
          </div>
        </div>
      </div>
    );
  };

  const renderToolChips = (contract: ChatResponse) => {
    const hasEvidence = contract.evidence && contract.evidence.length > 0;
    const hasTimeline = contract.timeline && contract.timeline.length > 0;

    return (
      <div className="tool-chips">
        {hasEvidence && (
          <button className="tool-chip" onClick={() => onSelectEvidence(contract)}>
            <Database size={12} /> {contract.evidence.length} Authoritative Observations
          </button>
        )}
        {hasTimeline && (
          <button className="tool-chip" onClick={() => onSelectEvidence(contract)}>
            <Calendar size={12} /> Timeline Generated
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="message-list">
      <div className="message-list-header">
        <h2>AI INVESTIGATION</h2>
      </div>

      <div className="messages-scroll">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.role === 'user' ? (
              <div className="message-content user-bubble">
                <span className="user-label">User</span>
                <p>{msg.content}</p>
              </div>
            ) : (
              <div className="message-content ai-bubble">
                {msg.responseContract && (
                  <ClaudeCompletedThought contract={msg.responseContract} />
                )}

                {msg.responseContract && (
                  <DetectionStatusBanner contract={msg.responseContract} />
                )}

                <div className="ai-response-text">
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>

                  {msg.responseContract?.evidence && msg.responseContract.evidence.some(e => e.crop_url) && (
                    <div className="inline-crops-container">
                      <div className="inline-crops-header-row">
                        <span className="inline-crops-header">📸 Verified Keyframe Evidence ({msg.responseContract.evidence.filter(e => e.crop_url).length})</span>
                        <span className="inline-crops-sub">Click any keyframe to focus in Evidence Panel</span>
                      </div>
                      <div className="inline-crops-grid">
                        {msg.responseContract.evidence
                          .filter(e => e.crop_url)
                          .slice(0, 8)
                          .map((ev, i) => (
                            <div 
                              key={i} 
                              className="inline-crop-card"
                              onClick={() => onSelectEvidence(msg.responseContract!)}
                              title={`View ${ev.person_id || 'evidence'} in Evidence Panel`}
                            >
                              <img src={ev.crop_url!} alt={ev.person_id || 'crop'} className="inline-crop-img" />
                              <span className="inline-crop-tag">{ev.person_id ? ev.person_id.replace('PERSON_', 'P_') : `#${i+1}`}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>

                {msg.responseContract && renderToolChips(msg.responseContract)}
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message assistant">
            <ClaudeAgentThinking />
          </div>
        )}
      </div>
    </div>
  );
}
