import React from 'react';
import { Globe, Brain, Target, Users, Github, ExternalLink } from 'lucide-react';

const About: React.FC = () => {
  const technologies = [
    { name: 'Python', description: 'Primary development language' },
    { name: 'Scikit-learn', description: 'Machine learning framework' },
    { name: 'TensorFlow', description: 'Deep learning capabilities' },
    { name: 'Pandas', description: 'Data manipulation and analysis' },
    { name: 'React', description: 'Frontend user interface' },
    { name: 'D3.js', description: 'Data visualization' }
  ];

  const datasets = [
    { name: 'World Bank Open Data', description: 'Economic and development indicators' },
    { name: 'UN Statistics', description: 'Official SDG tracking data' },
    { name: 'IEA Energy Data', description: 'Global energy consumption statistics' },
    { name: 'EDGAR Database', description: 'Emissions Database for Global Atmospheric Research' }
  ];

  const teamMembers = [
    { role: 'Data Scientist', focus: 'ML Model Development' },
    { role: 'Climate Researcher', focus: 'Domain expertise' },
    { role: 'Software Engineer', focus: 'Platform development' },
    { role: 'Ethics Advisor', focus: 'Responsible AI practices' }
  ];

  return (
    <section id="about" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            About the Solution
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            A comprehensive AI-driven approach to addressing UN SDG 13 through predictive modeling and data-driven insights.
          </p>
        </div>

        {/* Project Overview */}
        <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50 mb-12">
          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Project Overview</h3>
              <p className="text-slate-600 mb-6 leading-relaxed">
                ClimateAI is an advanced machine learning solution designed to forecast carbon emissions and support 
                evidence-based climate action. By analyzing historical data from 195 countries, our model provides 
                accurate predictions that help policymakers and organizations make informed decisions toward achieving 
                SDG 13: Climate Action.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg">
                  <Globe className="h-8 w-8 text-green-600 mb-2" />
                  <div className="text-2xl font-bold text-slate-900">195</div>
                  <div className="text-sm text-slate-600">Countries Analyzed</div>
                </div>
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg">
                  <Brain className="h-8 w-8 text-blue-600 mb-2" />
                  <div className="text-2xl font-bold text-slate-900">95%</div>
                  <div className="text-sm text-slate-600">Accuracy Rate</div>
                </div>
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
                  <Target className="h-8 w-8 text-purple-600 mb-2" />
                  <div className="text-2xl font-bold text-slate-900">2030</div>
                  <div className="text-sm text-slate-600">SDG Target Year</div>
                </div>
                <div className="bg-gradient-to-r from-pink-50 to-red-50 p-4 rounded-lg">
                  <Users className="h-8 w-8 text-pink-600 mb-2" />
                  <div className="text-2xl font-bold text-slate-900">12</div>
                  <div className="text-sm text-slate-600">Partner Organizations</div>
                </div>
              </div>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Technical Approach</h3>
              <div className="space-y-4">
                <div className="bg-gradient-to-r from-slate-50 to-blue-50 p-4 rounded-lg border border-slate-200/50">
                  <h4 className="font-semibold text-slate-900 mb-2">1. Data Collection & Preprocessing</h4>
                  <p className="text-sm text-slate-600">
                    Aggregated data from multiple authoritative sources, cleaned and normalized for consistency.
                  </p>
                </div>
                <div className="bg-gradient-to-r from-slate-50 to-blue-50 p-4 rounded-lg border border-slate-200/50">
                  <h4 className="font-semibold text-slate-900 mb-2">2. Feature Engineering</h4>
                  <p className="text-sm text-slate-600">
                    Created meaningful features including GDP ratios, energy intensity, and industrial activity indices.
                  </p>
                </div>
                <div className="bg-gradient-to-r from-slate-50 to-blue-50 p-4 rounded-lg border border-slate-200/50">
                  <h4 className="font-semibold text-slate-900 mb-2">3. Model Training</h4>
                  <p className="text-sm text-slate-600">
                    Random Forest Regression with hyperparameter tuning and cross-validation for optimal performance.
                  </p>
                </div>
                <div className="bg-gradient-to-r from-slate-50 to-blue-50 p-4 rounded-lg border border-slate-200/50">
                  <h4 className="font-semibold text-slate-900 mb-2">4. Validation & Deployment</h4>
                  <p className="text-sm text-slate-600">
                    Rigorous testing against real-world data with continuous monitoring and improvement.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          <div className="bg-white rounded-2xl p-6 shadow-xl border border-slate-200/50">
            <h3 className="text-xl font-bold text-slate-900 mb-4">Technology Stack</h3>
            <div className="space-y-3">
              {technologies.map((tech, index) => (
                <div key={index} className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                  <span className="font-medium text-slate-900">{tech.name}</span>
                  <span className="text-sm text-slate-600">{tech.description}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-xl border border-slate-200/50">
            <h3 className="text-xl font-bold text-slate-900 mb-4">Data Sources</h3>
            <div className="space-y-3">
              {datasets.map((dataset, index) => (
                <div key={index} className="p-3 bg-slate-50 rounded-lg">
                  <div className="font-medium text-slate-900 mb-1">{dataset.name}</div>
                  <div className="text-sm text-slate-600">{dataset.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Team & Collaboration */}
        <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50 mb-12">
          <h3 className="text-2xl font-bold text-slate-900 mb-6 text-center">Interdisciplinary Team</h3>
          <div className="grid md:grid-cols-4 gap-6">
            {teamMembers.map((member, index) => (
              <div key={index} className="text-center">
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-xl mb-4">
                  <Users className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                  <h4 className="font-semibold text-slate-900">{member.role}</h4>
                  <p className="text-sm text-slate-600 mt-1">{member.focus}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center">
          <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-2xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">Open Source & Collaboration</h3>
            <p className="text-lg opacity-90 mb-6 max-w-2xl mx-auto">
              Our solution is built on open science principles. We believe in transparency, collaboration, and 
              shared progress toward climate goals.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-white text-slate-900 px-6 py-3 rounded-xl font-semibold hover:bg-slate-100 transition-colors flex items-center justify-center">
                <Github className="h-5 w-5 mr-2" />
                View on GitHub
              </button>
              <button className="bg-white/10 backdrop-blur-sm text-white px-6 py-3 rounded-xl font-semibold hover:bg-white/20 transition-colors flex items-center justify-center">
                <ExternalLink className="h-5 w-5 mr-2" />
                Read Documentation
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;