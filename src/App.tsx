import React, { useState } from 'react';
import { TreePine, Activity, Target, Shield, Brain, TrendingUp, Users, Globe, ChevronDown } from 'lucide-react';
import Header from './components/Header';
import Hero from './components/Hero';
import ModelDemo from './components/ModelDemo';
import DataVisualization from './components/DataVisualization';
import Results from './components/Results';
import EthicalConsiderations from './components/EthicalConsiderations';
import About from './components/About';
import Footer from './components/Footer';

function App() {
  const [activeSection, setActiveSection] = useState('home');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      <main>
        <Hero />
        <ModelDemo />
        <DataVisualization />
        <Results />
        <EthicalConsiderations />
        <About />
      </main>
      <Footer />
    </div>
  );
}

export default App;