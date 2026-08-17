import { useState } from 'react';
import { Layout } from './components/Layout';
import { ChatBox } from './components/Chat/ChatBox';
import { MessageList } from './components/Chat/MessageList';
import { EvidencePanel } from './components/Evidence/EvidencePanel';
import { apiClient, ChatResponse } from './api/client';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  responseContract?: ChatResponse;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedContract, setSelectedContract] = useState<ChatResponse | null>(null);

  const handleSendMessage = async (query: string) => {
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // POST to backend API - Frontend does NO RAG logic
      const responseContract = await apiClient.chat(query);
      
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseContract.answer || 'No response provided.',
        responseContract
      };
      
      setMessages((prev) => [...prev, aiMsg]);
      setSelectedContract(responseContract); // Auto-select latest evidence
    } catch (error: any) {
      console.error(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: error?.message ? `Connection error: ${error.message}` : 'System error: Could not complete investigation.'
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="investigation-area">
        <MessageList 
          messages={messages} 
          isLoading={isLoading} 
          onSelectEvidence={setSelectedContract} 
        />
        <ChatBox onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
      <div className="evidence-sidebar">
        <EvidencePanel selectedContract={selectedContract} />
      </div>
    </Layout>
  );
}

export default App;
