import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { ViolationsPage } from './pages/ViolationsPage';
import { CamerasPage } from './pages/CamerasPage';
import { ModelStatusPage } from './pages/ModelStatusPage';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage />;
      case 'analysis':
        return <AnalysisPage />;
      case 'violations':
        return <ViolationsPage />;
      case 'cameras':
        return <CamerasPage />;
      case 'models':
        return <ModelStatusPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans selection:bg-sky-500 selection:text-white">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {renderContent()}
      </main>
      <footer className="border-t border-slate-800 bg-slate-950/40 py-4 text-center text-xs text-slate-500">
        Smart Traffic Violation Detection System &bull; Decision Support Platform &bull; Antigravity Multi-Agent Architecture
      </footer>
    </div>
  );
}

export default App;

