import React from 'react';
import { BarChart3, PieChart, Activity, Brain } from 'lucide-react';

interface DataVisualizationProps {
  analysisData: {
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
  predictions: {
    fake: number;
    real: number;
  };
}

const DataVisualization: React.FC<DataVisualizationProps> = ({ analysisData, predictions }) => {
  const getSentimentColor = (polarity: number) => {
    if (polarity > 0.1) return 'text-green-600 bg-green-100';
    if (polarity < -0.1) return 'text-red-600 bg-red-100';
    return 'text-gray-600 bg-gray-100';
  };

  const getSentimentLabel = (polarity: number) => {
    if (polarity > 0.1) return 'Positive';
    if (polarity < -0.1) return 'Negative';
    return 'Neutral';
  };

  const getSubjectivityColor = (subjectivity: number) => {
    if (subjectivity > 0.6) return 'text-purple-600 bg-purple-100';
    if (subjectivity > 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-blue-600 bg-blue-100';
  };

  const getSubjectivityLabel = (subjectivity: number) => {
    if (subjectivity > 0.6) return 'Subjective';
    if (subjectivity > 0.4) return 'Moderate';
    return 'Objective';
  };

  const getRiskLevel = (value: number, thresholds: { low: number; medium: number }) => {
    if (value <= thresholds.low) return 'low';
    if (value <= thresholds.medium) return 'medium';
    return 'high';
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-green-500';
      case 'medium': return 'bg-yellow-500';
      case 'high': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const capitalRiskLevel = getRiskLevel(analysisData.capital_ratio, { low: 0.02, medium: 0.05 });
  const punctuationRiskLevel = getRiskLevel(analysisData.punctuation_ratio, { low: 0.05, medium: 0.1 });
  const exclamationRiskLevel = getRiskLevel(analysisData.exclamation_count, { low: 2, medium: 5 });

  return (
    <div className="space-y-6">
      {/* Prediction Confidence Chart */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <PieChart className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Prediction Confidence</h3>
        </div>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-red-700">Fake News</span>
              <span className="text-sm text-gray-600">{(predictions.fake * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-red-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${predictions.fake * 100}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-green-700">Real News</span>
              <span className="text-sm text-gray-600">{(predictions.real * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-green-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${predictions.real * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Analysis */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="h-5 w-5 text-purple-600" />
          <h3 className="text-lg font-semibold">Sentiment Analysis</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className={`p-4 rounded-lg ${getSentimentColor(analysisData.sentiment_polarity)}`}>
            <div className="text-sm font-medium mb-1">Sentiment Polarity</div>
            <div className="text-2xl font-bold">{getSentimentLabel(analysisData.sentiment_polarity)}</div>
            <div className="text-xs opacity-75">{analysisData.sentiment_polarity.toFixed(3)}</div>
          </div>
          
          <div className={`p-4 rounded-lg ${getSubjectivityColor(analysisData.sentiment_subjectivity)}`}>
            <div className="text-sm font-medium mb-1">Subjectivity</div>
            <div className="text-2xl font-bold">{getSubjectivityLabel(analysisData.sentiment_subjectivity)}</div>
            <div className="text-xs opacity-75">{(analysisData.sentiment_subjectivity * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Risk Indicators */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="h-5 w-5 text-orange-600" />
          <h3 className="text-lg font-semibold">Risk Indicators</h3>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getRiskColor(capitalRiskLevel)}`} />
              <span className="text-sm font-medium">Capital Letters Ratio</span>
            </div>
            <div className="text-sm text-gray-600">
              {(analysisData.capital_ratio * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getRiskColor(punctuationRiskLevel)}`} />
              <span className="text-sm font-medium">Punctuation Ratio</span>
            </div>
            <div className="text-sm text-gray-600">
              {(analysisData.punctuation_ratio * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getRiskColor(exclamationRiskLevel)}`} />
              <span className="text-sm font-medium">Exclamation Count</span>
            </div>
            <div className="text-sm text-gray-600">
              {analysisData.exclamation_count}
            </div>
          </div>
        </div>
      </div>

      {/* Text Statistics */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="h-5 w-5 text-indigo-600" />
          <h3 className="text-lg font-semibold">Text Statistics</h3>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{analysisData.word_count}</div>
            <div className="text-xs text-gray-600">Words</div>
          </div>
          
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{analysisData.sentence_count}</div>
            <div className="text-xs text-gray-600">Sentences</div>
          </div>
          
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{analysisData.avg_word_length.toFixed(1)}</div>
            <div className="text-xs text-gray-600">Avg Word Length</div>
          </div>
          
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">{analysisData.question_count}</div>
            <div className="text-xs text-gray-600">Questions</div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Risk Level Legend</h4>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span>Low Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full" />
            <span>Medium Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span>High Risk</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataVisualization;