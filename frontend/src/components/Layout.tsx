import { ReactNode } from 'react';
import { Camera, Users, Clock, AlertTriangle, MessageSquare, Search } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="layout-container">
      {/* Sidebar Nav */}
      <nav className="sidebar">
        <div className="brand">
          <div className="status-indicator"></div>
          <span className="brand-text">VISTA AI</span>
        </div>
        
        <div className="nav-section">
          <h3 className="nav-heading">Dashboard</h3>
          <ul className="nav-list">
            <li className="nav-item"><Camera size={18}/> Cameras</li>
            <li className="nav-item"><Users size={18}/> Persons</li>
            <li className="nav-item"><AlertTriangle size={18}/> Events</li>
            <li className="nav-item"><Clock size={18}/> Timeline</li>
          </ul>
        </div>

        <div className="nav-section">
          <h3 className="nav-heading">AI</h3>
          <ul className="nav-list">
            <li className="nav-item active"><MessageSquare size={18}/> AI Chat</li>
            <li className="nav-item"><Search size={18}/> Search</li>
          </ul>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {children}
      </main>
    </div>
  );
}
