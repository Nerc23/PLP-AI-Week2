import React from 'react';
import { Shield, Users, AlertTriangle, CheckCircle } from 'lucide-react';

const EthicalConsiderations: React.FC = () => {
  const considerations = [
    {
      icon: Shield,
      title: "Data Privacy & Security",
      description: "All data sources are publicly available and anonymized. We implement strict data governance protocols.",
      status: "addressed",
      details: [
        "Only aggregated country-level data used",
        "No personal or sensitive information collected",
        "Transparent data sources from UN, World Bank, and official government statistics"
      ]
    },
    {
      icon: Users,
      title: "Algorithmic Bias",
      description: "We actively identify and mitigate potential biases in our model to ensure fair representation.",
      status: "monitoring",
      details: [
        "Equal representation of developing and developed nations",
        "Regular bias audits using fairness metrics",
        "Diverse training data spanning all continents and economic levels"
      ]
    },
    {
      icon: AlertTriangle,
      title: "Prediction Limitations",
      description: "Our model provides estimates, not absolute truths. Decisions should combine AI insights with human expertise.",
      status: "important",
      details: [
        "95% accuracy still means 5% uncertainty",
        "Black swan events (pandemics, wars) can't be predicted",
        "Model should complement, not replace, human judgment"
      ]
    },
    {
      icon: CheckCircle,
      title: "Positive Impact",
      description: "Our AI solution promotes transparency and supports evidence-based climate action worldwide.",
      status: "positive",
      details: [
        "Open-source methodology for reproducibility",
        "Supports climate justice and equitable policies",
        "Enables proactive rather than reactive climate action"
      ]
    }
  ];

  const principles = [
    {
      title: "Transparency",
      description: "All methodologies, data sources, and limitations are clearly documented and publicly available."
    },
    {
      title: "Fairness",
      description: "Equal consideration for all countries and regions, regardless of economic status or political alignment."
    },
    {
      title: "Accountability",
      description: "Clear ownership of model decisions with mechanisms for feedback and continuous improvement."
    },
    {
      title: "Sustainability",
      description: "The AI solution itself is designed to minimize computational resources and energy consumption."
    }
  ];

  return (
    <section id="ethics" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            Ethical Considerations
          </h2>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            Responsible AI development requires careful consideration of ethical implications, potential biases, and societal impact.
          </p>
        </div>

        {/* Ethical Considerations Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {considerations.map((consideration, index) => {
            const Icon = consideration.icon;
            const statusColors = {
              addressed: 'bg-green-100 text-green-800 border-green-200',
              monitoring: 'bg-blue-100 text-blue-800 border-blue-200',
              important: 'bg-yellow-100 text-yellow-800 border-yellow-200',
              positive: 'bg-purple-100 text-purple-800 border-purple-200'
            };

            return (
              <div key={index} className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-6 shadow-xl border border-slate-200/50">
                <div className="flex items-start mb-4">
                  <div className="bg-white p-3 rounded-xl shadow-lg mr-4">
                    <Icon className="h-6 w-6 text-slate-700" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-slate-900 mb-2">{consideration.title}</h3>
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium border ${statusColors[consideration.status]}`}>
                      {consideration.status.charAt(0).toUpperCase() + consideration.status.slice(1)}
                    </span>
                  </div>
                </div>
                
                <p className="text-slate-600 mb-4">{consideration.description}</p>
                
                <div className="space-y-2">
                  {consideration.details.map((detail, detailIndex) => (
                    <div key={detailIndex} className="flex items-start">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-slate-600">{detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Guiding Principles */}
        <div className="bg-gradient-to-r from-slate-900 to-blue-900 rounded-2xl p-8 text-white">
          <h3 className="text-2xl font-bold mb-6 text-center">Our Guiding Principles</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {principles.map((principle, index) => (
              <div key={index} className="text-center">
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 mb-4">
                  <h4 className="font-bold text-lg mb-2">{principle.title}</h4>
                  <p className="text-sm opacity-90">{principle.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bias Mitigation */}
        <div className="mt-16 grid lg:grid-cols-2 gap-12">
          <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-8 border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6">Bias Mitigation Strategies</h3>
            <div className="space-y-4">
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Data Representation</h4>
                <p className="text-sm text-slate-600">
                  Ensuring equal representation of all economic development levels and geographic regions in our training data.
                </p>
              </div>
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Regular Auditing</h4>
                <p className="text-sm text-slate-600">
                  Quarterly bias audits using established fairness metrics to identify and correct potential inequities.
                </p>
              </div>
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Diverse Perspectives</h4>
                <p className="text-sm text-slate-600">
                  Collaborative development with climate scientists, ethicists, and policy experts from around the world.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-8 border border-slate-200/50">
            <h3 className="text-2xl font-bold text-slate-900 mb-6">Continuous Improvement</h3>
            <div className="space-y-4">
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Stakeholder Feedback</h4>
                <p className="text-sm text-slate-600">
                  Regular consultation with climate organizations, policymakers, and affected communities.
                </p>
              </div>
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Model Updates</h4>
                <p className="text-sm text-slate-600">
                  Continuous learning and model refinement based on new data and changing climate patterns.
                </p>
              </div>
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h4 className="font-semibold text-slate-900 mb-2">Open Source</h4>
                <p className="text-sm text-slate-600">
                  Publishing our methodology and code for peer review and community contribution.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EthicalConsiderations;