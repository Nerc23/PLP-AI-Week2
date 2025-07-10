import React from 'react';
import { Globe, Target, TrendingUp, ChevronDown } from 'lucide-react';

const Hero: React.FC = () => {
  const scrollToDemo = () => {
    const element = document.getElementById('demo');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="home" className="pt-20 pb-16 min-h-screen flex items-center">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="flex justify-center space-x-4 mb-8">
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-4 rounded-full">
              <Globe className="h-8 w-8 text-white" />
            </div>
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 p-4 rounded-full">
              <Target className="h-8 w-8 text-white" />
            </div>
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 p-4 rounded-full">
              <TrendingUp className="h-8 w-8 text-white" />
            </div>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold text-slate-900 mb-6">
            AI for <span className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">Climate Action</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-slate-600 mb-8 max-w-4xl mx-auto leading-relaxed">
            Leveraging machine learning to forecast carbon emissions and drive sustainable development through 
            <span className="font-semibold text-green-600"> UN SDG 13: Climate Action</span>
          </p>
          
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 mb-12 max-w-4xl mx-auto border border-slate-200/50 shadow-xl">
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">2030</div>
                <div className="text-sm text-slate-600">Target Year</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">45%</div>
                <div className="text-sm text-slate-600">Emission Reduction Goal</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">95%</div>
                <div className="text-sm text-slate-600">Model Accuracy</div>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={scrollToDemo}
              className="bg-gradient-to-r from-green-600 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-green-700 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              Explore the Model
            </button>
            <a
              href="#about"
              className="bg-white text-slate-900 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-slate-50 transition-all duration-300 border border-slate-200 shadow-lg"
            >
              Learn More
            </a>
          </div>
        </div>
        
        <div className="flex justify-center mt-16 animate-bounce">
          <ChevronDown className="h-6 w-6 text-slate-400" />
        </div>
      </div>
    </section>
  );
};

export default Hero;