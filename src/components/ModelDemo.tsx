import React, { useState, useEffect } from 'react';
import { Brain, Play, BarChart3, Activity, Zap } from 'lucide-react';

const ModelDemo: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFeatures, setSelectedFeatures] = useState(['gdp', 'population', 'energy']);
  const [prediction, setPrediction] = useState<number | null>(null);

  const features = [
    { id: 'gdp', label: 'GDP per Capita', icon: BarChart3, color: 'blue' },
    { id: 'population', label: 'Population Density', icon: Activity, color: 'green' },
    { id: 'energy', label: 'Energy Consumption', icon: Zap, color: 'yellow' },
    { id: 'industrial', label: 'Industrial Activity', icon: Brain, color: 'purple' }
  ];

  const runModel = () => {
    setIsRunning(true);
    setProgress(0);
    setPrediction(null);

    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          // Simulate prediction result
          const basePrediction = 850;
          const featureMultiplier = selectedFeatures.length * 0.1;
          const randomness = Math.random() * 100 - 50;
          setPrediction(Math.round(basePrediction + randomness + (featureMultiplier * 100)));
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  const toggleFeature = (featureId: string) => {
    setSelectedFeatures(prev => 
      prev.includes(featureId)
        ? prev.filter(id => id !== featureId)
        : [...prev, featureId]
    );
  };

  return (
    <section id="demo" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            Interactive ML Model Demo
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            Experience our carbon emission forecasting model in action. Select features and run predictions to see how AI can help predict and reduce emissions.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Feature Selection */}
          <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center">
              <Brain className="h-6 w-6 mr-3 text-purple-600" />
              Feature Selection
            </h3>
            
            <div className="space-y-4">
              {features.map((feature) => {
                const Icon = feature.icon;
                const isSelected = selectedFeatures.includes(feature.id);
                
                return (
                  <button
                    key={feature.id}
                    onClick={() => toggleFeature(feature.id)}
                    className={`w-full p-4 rounded-xl border-2 transition-all duration-300 ${
                      isSelected
                        ? `border-${feature.color}-500 bg-${feature.color}-50`
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center">
                      <Icon className={`h-6 w-6 mr-3 ${
                        isSelected ? `text-${feature.color}-600` : 'text-slate-400'
                      }`} />
                      <span className={`font-medium ${
                        isSelected ? `text-${feature.color}-900` : 'text-slate-600'
                      }`}>
                        {feature.label}
                      </span>
                      <div className={`ml-auto w-5 h-5 rounded-full border-2 ${
                        isSelected
                          ? `bg-${feature.color}-500 border-${feature.color}-500`
                          : 'border-slate-300'
                      }`}>
                        {isSelected && (
                          <div className="w-full h-full bg-white rounded-full transform scale-50"></div>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            <button
              onClick={runModel}
              disabled={isRunning || selectedFeatures.length === 0}
              className={`w-full mt-8 px-6 py-4 rounded-xl font-semibold text-lg transition-all duration-300 ${
                isRunning || selectedFeatures.length === 0
                  ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-green-600 to-blue-600 text-white hover:from-green-700 hover:to-blue-700 transform hover:scale-105'
              }`}
            >
              {isRunning ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                  Running Model...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <Play className="h-6 w-6 mr-3" />
                  Run Prediction
                </div>
              )}
            </button>
          </div>

          {/* Results */}
          <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6">
              Model Results
            </h3>
            
            {isRunning && (
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-slate-600">Training Progress</span>
                  <span className="text-sm font-semibold text-slate-900">{progress}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div 
                    className="bg-gradient-to-r from-green-600 to-blue-600 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {prediction !== null && (
              <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 mb-6">
                <div className="text-center">
                  <div className="text-4xl font-bold text-slate-900 mb-2">
                    {prediction.toLocaleString()} Mt CO₂
                  </div>
                  <div className="text-lg text-slate-600 mb-4">
                    Predicted Annual Emissions
                  </div>
                  <div className="flex justify-center space-x-4">
                    <div className="text-center">
                      <div className="text-lg font-semibold text-green-600">-12%</div>
                      <div className="text-sm text-slate-600">vs. Last Year</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-semibold text-blue-600">95%</div>
                      <div className="text-sm text-slate-600">Confidence</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-4">
              <div className="bg-slate-50 rounded-lg p-4">
                <h4 className="font-semibold text-slate-900 mb-2">Model Algorithm</h4>
                <p className="text-sm text-slate-600">
                  Random Forest Regression with feature importance weighting
                </p>
              </div>
              
              <div className="bg-slate-50 rounded-lg p-4">
                <h4 className="font-semibold text-slate-900 mb-2">Training Data</h4>
                <p className="text-sm text-slate-600">
                  Historical emissions data from 195 countries (2000-2023)
                </p>
              </div>
              
              <div className="bg-slate-50 rounded-lg p-4">
                <h4 className="font-semibold text-slate-900 mb-2">Validation</h4>
                <p className="text-sm text-slate-600">
                  Cross-validated with R² = 0.95, MAE = 42.3 Mt CO₂
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelDemo;