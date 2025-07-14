import React, { useState } from 'react';
import { Shield, Brain, BarChart3, Settings, Info } from 'lucide-react';
import NewsAnalyzer from './components/NewsAnalyzer';
import ResultsDisplay from './components/ResultsDisplay';
import DataVisualization from './components/DataVisualization';

function App() {
  const [activeTab, setActiveTab] = useState<'analyze' | 'results' | 'visualization' | 'about'>('analyze');
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const tabs = [
    { id: 'analyze', label: 'Analyze', icon: Shield },
    { id: 'results', label: 'Results', icon: BarChart3 },
    { id: 'visualization', label: 'Visualization', icon: Brain },
    { id: 'about', label: 'About', icon: Info }
  ];

  const sampleResults = {
    prediction: {
      label: 'Real',
      confidence: 0.85,
      probabilities: {
        fake: 0.15,
        real: 0.85
      }
    },
    analysis: {
      word_count: 324,
      sentence_count: 18,
      avg_word_length: 5.2,
      sentiment_polarity: 0.1,
      sentiment_subjectivity: 0.3,
      exclamation_count: 1,
      question_count: 2,
      capital_ratio: 0.02,
      punctuation_ratio: 0.08
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Fake News Detection</h1>
                <p className="text-sm text-gray-600">AI-Powered News Verification</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                System Active
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-1 py-4 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'analyze' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                News Article Analysis
              </h2>
              <p className="text-lg text-gray-600">
                Analyze news articles for potential misinformation using advanced AI algorithms
              </p>
            </div>
            <NewsAnalyzer />
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Analysis Results
              </h2>
              <p className="text-lg text-gray-600">
                Detailed breakdown of the fake news detection analysis
              </p>
            </div>
            <ResultsDisplay results={sampleResults} />
          </div>
        )}

        {activeTab === 'visualization' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Data Visualization
              </h2>
              <p className="text-lg text-gray-600">
                Visual representation of analysis metrics and patterns
              </p>
            </div>
            <DataVisualization 
              analysisData={sampleResults.analysis} 
              predictions={sampleResults.prediction.probabilities}
            />
          </div>
        )}

        {activeTab === 'about' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                About Fake News Detection
              </h2>
              <p className="text-lg text-gray-600">
                Understanding our AI-powered approach to identifying misinformation
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Brain className="h-5 w-5 text-blue-600" />
                  How It Works
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• <strong>NLP Processing:</strong> Advanced text analysis and feature extraction</li>
                  <li>• <strong>Sentiment Analysis:</strong> Emotional tone and bias detection</li>
                  <li>• <strong>Linguistic Patterns:</strong> Writing style and structure analysis</li>
                  <li>• <strong>Source Verification:</strong> Credibility assessment algorithms</li>
                  <li>• <strong>Machine Learning:</strong> Trained on thousands of verified articles</li>
                </ul>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Settings className="h-5 w-5 text-green-600" />
                  Key Features
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• <strong>Real-time Analysis:</strong> Instant fake news detection</li>
                  <li>• <strong>Confidence Scoring:</strong> Probability-based results</li>
                  <li>• <strong>Detailed Insights:</strong> Comprehensive analysis reports</li>
                  <li>• <strong>Multiple Inputs:</strong> Text and URL analysis support</li>
                  <li>• <strong>Visual Reports:</strong> Interactive charts and metrics</li>
                </ul>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-purple-600" />
                  Analysis Metrics
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• <strong>Linguistic Features:</strong> Word patterns, sentence structure</li>
                  <li>• <strong>Emotional Indicators:</strong> Sentiment and subjectivity</li>
                  <li>• <strong>Structural Analysis:</strong> Article length and formatting</li>
                  <li>• <strong>Credibility Signals:</strong> Source reliability factors</li>
                  <li>• <strong>Risk Assessment:</strong> Multi-factor scoring system</li>
                </ul>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Info className="h-5 w-5 text-orange-600" />
                  Important Notes
                </h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• <strong>Tool Assistance:</strong> This is a detection aid, not definitive proof</li>
                  <li>• <strong>Human Verification:</strong> Always cross-check with multiple sources</li>
                  <li>• <strong>Context Matters:</strong> Consider broader context and source reputation</li>
                  <li>• <strong>Continuous Learning:</strong> Model improves with more data</li>
                  <li>• <strong>False Positives:</strong> Some legitimate articles may be flagged</li>
                </ul>
              </div>
            </div>
            
            <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
              <h3 className="text-lg font-semibold mb-3 text-blue-900">
                Technology Stack
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="bg-white p-3 rounded border">
                  <div className="font-medium text-gray-900">Frontend</div>
                  <div className="text-gray-600">React, TypeScript</div>
                </div>
                <div className="bg-white p-3 rounded border">
                  <div className="font-medium text-gray-900">Backend</div>
                  <div className="text-gray-600">Python, Flask</div>
                </div>
                <div className="bg-white p-3 rounded border">
                  <div className="font-medium text-gray-900">ML/AI</div>
                  <div className="text-gray-600">Scikit-learn, NLTK</div>
                </div>
                <div className="bg-white p-3 rounded border">
                  <div className="font-medium text-gray-900">Data</div>
                  <div className="text-gray-600">BeautifulSoup, Pandas</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">Fake News Detection</h3>
              <p className="text-gray-400 text-sm">
                AI-powered system for identifying misinformation and fake news articles.
                Helping users make informed decisions about news credibility.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Features</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Real-time analysis</li>
                <li>Confidence scoring</li>
                <li>Detailed reports</li>
                <li>Visual insights</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Resources</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Documentation</li>
                <li>API Reference</li>
                <li>Model Training</li>
                <li>Support</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>&copy; 2025 Fake News Detection System. Built with modern AI and machine learning.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;