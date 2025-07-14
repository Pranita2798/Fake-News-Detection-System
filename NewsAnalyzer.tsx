import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Loader2, FileText, ExternalLink, AlertTriangle } from 'lucide-react';

interface AnalysisResult {
  prediction: {
    label: string;
    confidence: number;
    probabilities: {
      fake: number;
      real: number;
    };
  };
  analysis: {
    word_count: number;
    sentence_count: number;
    avg_word_length: number;
    sentiment_polarity: number;
    sentiment_subjectivity: number;
    exclamation_count: number;
    question_count: number;
    capital_ratio: number;
    punctuation_ratio: number;
  };
  timestamp: string;
}

const NewsAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'text' | 'url'>('text');

  const analyzeText = async (text: string) => {
    setIsAnalyzing(true);
    setError('');
    setResult(null);

    try {
      // Simulate API call with realistic fake news detection
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult: AnalysisResult = {
        prediction: {
          label: text.includes('!!!') || text.includes('SHOCKING') || text.includes('BREAKING') ? 'Fake' : 'Real',
          confidence: Math.random() * 0.3 + 0.7,
          probabilities: {
            fake: text.includes('!!!') ? Math.random() * 0.3 + 0.7 : Math.random() * 0.3 + 0.1,
            real: text.includes('!!!') ? Math.random() * 0.3 + 0.1 : Math.random() * 0.3 + 0.7
          }
        },
        analysis: {
          word_count: text.split(' ').length,
          sentence_count: text.split('.').length - 1,
          avg_word_length: text.split(' ').reduce((sum, word) => sum + word.length, 0) / text.split(' ').length,
          sentiment_polarity: Math.random() * 2 - 1,
          sentiment_subjectivity: Math.random(),
          exclamation_count: (text.match(/!/g) || []).length,
          question_count: (text.match(/\?/g) || []).length,
          capital_ratio: (text.match(/[A-Z]/g) || []).length / text.length,
          punctuation_ratio: (text.match(/[.,!?;:]/g) || []).length / text.length
        },
        timestamp: new Date().toISOString()
      };

      setResult(mockResult);
    } catch (err) {
      setError('Failed to analyze text. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyze = () => {
    const text = activeTab === 'text' ? inputText : inputUrl;
    if (!text.trim()) {
      setError('Please enter some text or URL to analyze.');
      return;
    }

    if (activeTab === 'url') {
      // In a real implementation, this would fetch the article content
      analyzeText(`Sample article content from ${inputUrl}`);
    } else {
      analyzeText(text);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <FileText className="h-8 w-8 text-blue-600" />
        <h2 className="text-2xl font-bold text-gray-800">News Analyzer</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex mb-6 border-b border-gray-200">
        <button
          onClick={() => setActiveTab('text')}
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'text'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Analyze Text
        </button>
        <button
          onClick={() => setActiveTab('url')}
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'url'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Analyze URL
        </button>
      </div>

      {/* Input Section */}
      <div className="mb-6">
        {activeTab === 'text' ? (
          <div>
            <label htmlFor="article-text" className="block text-sm font-medium text-gray-700 mb-2">
              Article Text
            </label>
            <textarea
              id="article-text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste the news article text here..."
              className="w-full h-40 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            />
          </div>
        ) : (
          <div>
            <label htmlFor="article-url" className="block text-sm font-medium text-gray-700 mb-2">
              Article URL
            </label>
            <div className="flex gap-2">
              <input
                id="article-url"
                type="url"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="https://example.com/news-article"
                className="flex-1 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <ExternalLink className="h-5 w-5 text-gray-400 self-center" />
            </div>
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={isAnalyzing}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-md font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          'Analyze Article'
        )}
      </button>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-red-600" />
          <span className="text-red-800">{error}</span>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mt-6 space-y-6">
          {/* Prediction Result */}
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              {result.prediction.label === 'Real' ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-red-600" />
              )}
              Prediction Result
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${result.prediction.label === 'Real' ? 'text-green-600' : 'text-red-600'}`}>
                  {result.prediction.label}
                </div>
                <div className="text-sm text-gray-600">Classification</div>
              </div>
              
              <div className="text-center">
                <div className={`text-2xl font-bold ${getConfidenceColor(result.prediction.confidence)}`}>
                  {(result.prediction.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Confidence</div>
              </div>
              
              <div className="text-center">
                <div className={`text-2xl font-bold ${getConfidenceColor(result.prediction.confidence)}`}>
                  {getConfidenceText(result.prediction.confidence)}
                </div>
                <div className="text-sm text-gray-600">Reliability</div>
              </div>
            </div>
            
            {/* Probability Bars */}
            <div className="mt-4 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-red-600 w-12">Fake:</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full"
                    style={{ width: `${result.prediction.probabilities.fake * 100}%` }}
                  />
                </div>
                <span className="text-sm text-gray-600">{(result.prediction.probabilities.fake * 100).toFixed(1)}%</span>
              </div>
              
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-green-600 w-12">Real:</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${result.prediction.probabilities.real * 100}%` }}
                  />
                </div>
                <span className="text-sm text-gray-600">{(result.prediction.probabilities.real * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Analysis Details */}
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">Detailed Analysis</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{result.analysis.word_count}</div>
                <div className="text-sm text-gray-600">Words</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{result.analysis.sentence_count}</div>
                <div className="text-sm text-gray-600">Sentences</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{result.analysis.avg_word_length.toFixed(1)}</div>
                <div className="text-sm text-gray-600">Avg Word Length</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{result.analysis.exclamation_count}</div>
                <div className="text-sm text-gray-600">Exclamations</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{result.analysis.question_count}</div>
                <div className="text-sm text-gray-600">Questions</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-blue-600">{(result.analysis.capital_ratio * 100).toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Capital Letters</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className={`text-2xl font-bold ${result.analysis.sentiment_polarity > 0 ? 'text-green-600' : result.analysis.sentiment_polarity < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                  {result.analysis.sentiment_polarity > 0 ? 'Positive' : result.analysis.sentiment_polarity < 0 ? 'Negative' : 'Neutral'}
                </div>
                <div className="text-sm text-gray-600">Sentiment</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-purple-600">{(result.analysis.sentiment_subjectivity * 100).toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Subjectivity</div>
              </div>
              
              <div className="bg-white p-4 rounded border">
                <div className="text-2xl font-bold text-orange-600">{(result.analysis.punctuation_ratio * 100).toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Punctuation</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewsAnalyzer;