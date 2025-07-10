import React from 'react';
import { TreePine, Github, ExternalLink, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  const sdgGoals = [
    'No Poverty', 'Zero Hunger', 'Good Health', 'Quality Education', 
    'Gender Equality', 'Clean Water', 'Affordable Energy', 'Decent Work',
    'Industry Innovation', 'Reduced Inequalities', 'Sustainable Cities', 
    'Responsible Consumption', 'Climate Action', 'Life Below Water',
    'Life on Land', 'Peace & Justice', 'Partnerships'
  ];

  return (
    <footer className="bg-slate-900 text-white py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="bg-gradient-to-r from-green-500 to-blue-600 p-2 rounded-xl">
                <TreePine className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold">ClimateAI</h3>
                <p className="text-sm text-slate-400">SDG 13 Solution</p>
              </div>
            </div>
            <p className="text-slate-300 mb-6 leading-relaxed">
              Leveraging artificial intelligence to forecast carbon emissions and support evidence-based 
              climate action in alignment with UN Sustainable Development Goal 13.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="bg-slate-800 p-3 rounded-lg hover:bg-slate-700 transition-colors">
                <Github className="h-5 w-5" />
              </a>
              <a href="#" className="bg-slate-800 p-3 rounded-lg hover:bg-slate-700 transition-colors">
                <ExternalLink className="h-5 w-5" />
              </a>
              <a href="#" className="bg-slate-800 p-3 rounded-lg hover:bg-slate-700 transition-colors">
                <Mail className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Research Paper</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Dataset</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">API Access</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Community</a></li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h4 className="font-semibold mb-4">Connect</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Partner Organizations</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Climate Scientists</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Policy Makers</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Developers</a></li>
              <li><a href="#" className="text-slate-400 hover:text-white transition-colors">Researchers</a></li>
            </ul>
          </div>
        </div>

        {/* UN SDG Goals */}
        <div className="mt-12 pt-8 border-t border-slate-800">
          <h4 className="font-semibold mb-4 text-center">UN Sustainable Development Goals</h4>
          <div className="flex flex-wrap justify-center gap-2">
            {sdgGoals.map((goal, index) => (
              <span 
                key={index} 
                className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                  index === 12 // Climate Action (SDG 13)
                    ? 'bg-green-600 text-white border-green-600'
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-600'
                }`}
              >
                {index + 1}. {goal}
              </span>
            ))}
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 pt-8 border-t border-slate-800 text-center text-sm text-slate-400">
          <p>© 2024 ClimateAI. Built for PLP Academy SDG Assignment. Open source and available for climate action.</p>
          <p className="mt-2">"AI can be the bridge between innovation and sustainability." — UN Tech Envoy</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;