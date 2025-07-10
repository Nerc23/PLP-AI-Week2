import React, { useState } from 'react';
import { TrendingUp, TrendingDown, BarChart3, PieChart } from 'lucide-react';

const DataVisualization: React.FC = () => {
  const [selectedChart, setSelectedChart] = useState('emissions');

  const emissionsData = [
    { year: 2018, emissions: 1200, target: 1150 },
    { year: 2019, emissions: 1180, target: 1100 },
    { year: 2020, emissions: 1050, target: 1050 },
    { year: 2021, emissions: 1120, target: 1000 },
    { year: 2022, emissions: 1080, target: 950 },
    { year: 2023, emissions: 1000, target: 900 }
  ];

  const sectorData = [
    { sector: 'Energy', percentage: 42, color: 'bg-red-500' },
    { sector: 'Transport', percentage: 28, color: 'bg-orange-500' },
    { sector: 'Industry', percentage: 18, color: 'bg-yellow-500' },
    { sector: 'Agriculture', percentage: 12, color: 'bg-green-500' }
  ];

  const regionData = [
    { region: 'Asia', value: 850, trend: '+2.1%' },
    { region: 'North America', value: 420, trend: '-3.2%' },
    { region: 'Europe', value: 380, trend: '-5.1%' },
    { region: 'Africa', value: 180, trend: '+1.8%' },
    { region: 'South America', value: 120, trend: '-0.5%' },
    { region: 'Oceania', value: 50, trend: '-1.2%' }
  ];

  return (
    <section id="visualization" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            Data Insights & Visualizations
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            Explore comprehensive data analysis that powers our AI model predictions and reveals critical emission patterns.
          </p>
        </div>

        <div className="mb-8">
          <div className="flex flex-wrap justify-center gap-4">
            <button
              onClick={() => setSelectedChart('emissions')}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                selectedChart === 'emissions'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <TrendingUp className="h-5 w-5 inline mr-2" />
              Emissions Trends
            </button>
            <button
              onClick={() => setSelectedChart('sectors')}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                selectedChart === 'sectors'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <PieChart className="h-5 w-5 inline mr-2" />
              Sector Breakdown
            </button>
            <button
              onClick={() => setSelectedChart('regions')}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                selectedChart === 'regions'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <BarChart3 className="h-5 w-5 inline mr-2" />
              Regional Analysis
            </button>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Chart */}
          <div className="lg:col-span-2">
            <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-8 shadow-xl border border-slate-200/50">
              {selectedChart === 'emissions' && (
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">
                    Global Emissions vs. SDG Targets
                  </h3>
                  <div className="relative h-64">
                    {emissionsData.map((data, index) => (
                      <div key={data.year} className="absolute inset-0 flex items-end">
                        <div 
                          className="w-full flex justify-center"
                          style={{ left: `${(index / (emissionsData.length - 1)) * 100}%`, transform: 'translateX(-50%)' }}
                        >
                          <div className="text-center">
                            <div className="flex items-end space-x-2">
                              <div 
                                className="bg-blue-500 rounded-t-lg w-8"
                                style={{ height: `${(data.emissions / 1200) * 200}px` }}
                              ></div>
                              <div 
                                className="bg-green-500 rounded-t-lg w-8"
                                style={{ height: `${(data.target / 1200) * 200}px` }}
                              ></div>
                            </div>
                            <div className="text-xs text-slate-600 mt-2">{data.year}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-center space-x-6 mt-6">
                    <div className="flex items-center">
                      <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
                      <span className="text-sm text-slate-600">Actual Emissions</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
                      <span className="text-sm text-slate-600">SDG Targets</span>
                    </div>
                  </div>
                </div>
              )}

              {selectedChart === 'sectors' && (
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">
                    Emissions by Sector
                  </h3>
                  <div className="space-y-4">
                    {sectorData.map((sector, index) => (
                      <div key={sector.sector} className="relative">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-slate-900">{sector.sector}</span>
                          <span className="text-sm text-slate-600">{sector.percentage}%</span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-4">
                          <div 
                            className={`${sector.color} h-4 rounded-full transition-all duration-1000`}
                            style={{ width: `${sector.percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedChart === 'regions' && (
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">
                    Regional Emission Patterns
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    {regionData.map((region) => (
                      <div key={region.region} className="bg-white rounded-xl p-4 border border-slate-200">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-semibold text-slate-900">{region.region}</h4>
                          <span className={`text-sm px-2 py-1 rounded ${
                            region.trend.startsWith('+') 
                              ? 'bg-red-100 text-red-700' 
                              : 'bg-green-100 text-green-700'
                          }`}>
                            {region.trend}
                          </span>
                        </div>
                        <div className="text-2xl font-bold text-slate-900">
                          {region.value} Mt CO₂
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Statistics Panel */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 shadow-xl border border-slate-200/50">
              <h3 className="text-xl font-bold text-slate-900 mb-4">Key Statistics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Global Emissions 2023</span>
                  <span className="font-bold text-slate-900">36.7 Gt CO₂</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Year-over-Year Change</span>
                  <span className="font-bold text-red-600">+1.1%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Required Reduction</span>
                  <span className="font-bold text-green-600">-45% by 2030</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-xl border border-slate-200/50">
              <h3 className="text-xl font-bold text-slate-900 mb-4">Model Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Accuracy Score</span>
                  <span className="font-bold text-green-600">95.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Mean Absolute Error</span>
                  <span className="font-bold text-slate-900">42.3 Mt CO₂</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">R² Score</span>
                  <span className="font-bold text-blue-600">0.95</span>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl p-6 border border-green-200/50">
              <h3 className="text-lg font-bold text-slate-900 mb-2">Impact Potential</h3>
              <p className="text-sm text-slate-600">
                Our AI model can help reduce prediction errors by 78% compared to traditional methods, enabling more effective climate policies.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DataVisualization;