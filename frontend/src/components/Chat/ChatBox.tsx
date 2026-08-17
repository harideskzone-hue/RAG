import { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatBoxProps {
  onSendMessage: (msg: string) => void;
  isLoading: boolean;
}

export function ChatBox({ onSendMessage, isLoading }: ChatBoxProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSendMessage(query);
      setQuery('');
    }
  };

  return (
    <div className="chat-box-container">
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          className="chat-input"
          placeholder="Ask VISTA... ➤"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="chat-submit" disabled={!query.trim() || isLoading}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
