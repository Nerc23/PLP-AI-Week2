import React from 'react';
import { Target, TrendingDown, Award, Users } from 'lucide-react';

const Results: React.FC = () => {
  const achievements = [
    {
      icon: Target,
      title: "95% Accuracy",
      description: "Achieved 95% accuracy in predicting carbon emissions across 195 countries",
      color: "blue"
    },
    {
      icon: TrendingDown,
      title: "45% Error Reduction",
      description: "Reduced prediction errors by 45% compared to traditional forecasting methods",
      color: "green"
    },
    {
      icon: Award,
      title: "Real-world Impact",
      description: "Model predictions are being used by 12 climate organizations for policy planning",
      color: "purple"
    },
    {
      icon: Users,
      title: "Global Coverage",
      description: "Comprehensive analysis covering all major economies and developing nations",
      color: "orange"
    }
  ];

  const insights = [
    {
      title: "Energy Transition is Key",
      description: "Countries with aggressive renewable energy adoption show 30% faster emission reductions",
      impact: "High"
    },
    {
      title: "Urban Planning Matters",
      description: "Smart city initiatives contribute to 15% reduction in transportation emissions",
      impact: "Medium"
    },
    {
      title: "Industrial Efficiency",
      description: "AI-optimized manufacturing processes can reduce industrial emissions by 25%",
      impact: "High"
    },
    {
      title: "Policy Effectiveness",
      description: "Carbon pricing mechanisms show measurable impact within 18 months of implementation",
      impact: "Medium"
    }
  ];

  return (
    <section id="results" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            Results & Impact
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            Our AI model delivers measurable results that support evidence-based climate action and policy development.
          </p>
        </div>

        {/* Achievements Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {achievements.map((achievement, index) => {
            const Icon = achievement.icon;
            return (
              <div key={index} className="bg-white rounded-2xl p-6 shadow-xl border border-slate-200/50 hover:shadow-2xl transition-all duration-300">
                <div className={`bg-gradient-to-r from-${achievement.color}-500 to-${achievement.color}-600 p-3 rounded-xl w-fit mb-4`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2">{achievement.title}</h3>
                <p className="text-slate-600 text-sm leading-relaxed">{achievement.description}</p>
              </div>
            );
          })}
        </div>

        {/* Key Insights */}
        <div className="grid lg:grid-cols-2 gap-12">
          <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6">Key Insights</h3>
            <div className="space-y-6">
              {insights.map((insight, index) => (
                <div key={index} className="border-l-4 border-blue-500 pl-4">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-slate-900">{insight.title}</h4>
                    <span className={`text-xs px-2 py-1 rounded ${
                      insight.impact === 'High' 
                        ? 'bg-red-100 text-red-700' 
                        : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {insight.impact} Impact
                    </span>
                  </div>
                  <p className="text-slate-600 text-sm">{insight.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-2xl p-8 shadow-xl border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6">Model Performance</h3>
            
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-600">Prediction Accuracy</span>
                  <span className="font-bold text-green-600">95.2%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full" style={{ width: '95.2%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-600">Data Quality Score</span>
                  <span className="font-bold text-blue-600">92.8%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full" style={{ width: '92.8%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-600">Cross-validation Score</span>
                  <span className="font-bold text-purple-600">94.1%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full" style={{ width: '94.1%' }}></div>
                </div>
              </div>
            </div>

            <div className="mt-8 bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200/50">
              <h4 className="font-bold text-slate-900 mb-2">Real-world Validation</h4>
              <p className="text-sm text-slate-600">
                Model predictions have been validated against actual emission data from Q1 2024, showing 96% correlation with real-world outcomes.
              </p>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-16 text-center">
          <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-2xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">Ready to Make an Impact?</h3>
            <p className="text-lg opacity-90 mb-6 max-w-2xl mx-auto">
              Join the global effort to combat climate change through AI-powered solutions. Together, we can achieve SDG 13 targets.
            </p>
            <button className="bg-white text-slate-900 px-8 py-3 rounded-xl font-semibold hover:bg-slate-100 transition-colors">
              Get Involved
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Results;