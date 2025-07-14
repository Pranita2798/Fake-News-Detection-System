import React from 'react';
import { Shield, AlertTriangle, TrendingUp, Eye, Brain, Target } from 'lucide-react';

interface ResultsDisplayProps {
  results: {
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
      sentiment_polarity: number;
      sentiment_subjectivity: number;
      exclamation_count: number;
      question_count: number;
      capital_ratio: number;
      punctuation_ratio: number;
    };
  };
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results }) => {
  const { prediction, analysis } = results;

  const getWarningLevel = (confidence: number) => {
    if (confidence >= 0.8) return 'low';
    if (confidence >= 0.6) return 'medium';
    return 'high';
  };

  const warningLevel = getWarningLevel(prediction.confidence);

  const getWarningColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getWarningIcon = (level: string) => {
    switch (level) {
      case 'low': return <Shield className="h-5 w-5" />;
      case 'medium': return <Eye className="h-5 w-5" />;
      case 'high': return <AlertTriangle className="h-5 w-5" />;
      default: return <Brain className="h-5 w-5" />;
    }
  };

  const getWarningMessage = (level: string, label: string) => {
    if (label === 'Real') {
      switch (level) {
        case 'low': return 'This article appears to be legitimate news.';
        case 'medium': return 'This article is likely real, but verify sources.';
        case 'high': return 'This article may be real, but requires fact-checking.';
        default: return 'Analysis inconclusive.';
      }
    } else {
      switch (level) {
        case 'low': return 'This article is likely fake or misleading.';
        case 'medium': return 'This article shows signs of being fake news.';
        case 'high': return 'This article is very likely fake or heavily biased.';
        default: return 'Analysis inconclusive.';
      }
    }
  };

  const getRiskFactors = () => {
    const factors = [];
    
    if (analysis.exclamation_count > 3) {
      factors.push('Excessive exclamation marks');
    }
    if (analysis.capital_ratio > 0.05) {
      factors.push('High proportion of capital letters');
    }
    if (analysis.sentiment_subjectivity > 0.7) {
      factors.push('Highly subjective language');
    }
    if (analysis.punctuation_ratio > 0.1) {
      factors.push('Excessive punctuation');
    }
    if (analysis.word_count < 100) {
      factors.push('Very short article');
    }
    if (Math.abs(analysis.sentiment_polarity) > 0.5) {
      factors.push('Emotionally charged content');
    }

    return factors;
  };

  const getPositiveFactors = () => {
    const factors = [];
    
    if (analysis.exclamation_count <= 2) {
      factors.push('Reasonable use of exclamations');
    }
    if (analysis.capital_ratio <= 0.03) {
      factors.push('Professional capitalization');
    }
    if (analysis.sentiment_subjectivity <= 0.4) {
      factors.push('Objective language');
    }
    if (analysis.word_count >= 200) {
      factors.push('Substantial content');
    }
    if (Math.abs(analysis.sentiment_polarity) <= 0.3) {
      factors.push('Balanced sentiment');
    }

    return factors;
  };

  const riskFactors = getRiskFactors();
  const positiveFactors = getPositiveFactors();

  return (
    <div className="space-y-6">
      {/* Main Alert */}
      <div className={`p-4 rounded-lg border ${getWarningColor(warningLevel)}`}>
        <div className="flex items-center gap-3 mb-2">
          {getWarningIcon(warningLevel)}
          <h3 className="font-semibold">
            {prediction.label === 'Real' ? 'Legitimate News' : 'Potential Fake News'}
          </h3>
        </div>
        <p className="text-sm">
          {getWarningMessage(warningLevel, prediction.label)}
        </p>
      </div>

      {/* Confidence Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg border shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <Target className="h-5 w-5 text-blue-600" />
            <h4 className="font-semibold">Confidence Score</h4>
          </div>
          <div className="text-3xl font-bold text-blue-600 mb-2">
            {(prediction.confidence * 100).toFixed(1)}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full"
              style={{ width: `${prediction.confidence * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-5 w-5 text-purple-600" />
            <h4 className="font-semibold">Reliability Index</h4>
          </div>
          <div className="text-3xl font-bold text-purple-600 mb-2">
            {warningLevel === 'low' ? 'High' : warningLevel === 'medium' ? 'Medium' : 'Low'}
          </div>
          <div className="text-sm text-gray-600">
            Based on linguistic analysis
          </div>
        </div>
      </div>

      {/* Risk Factors */}
      {riskFactors.length > 0 && (
        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
          <h4 className="font-semibold text-red-800 mb-2 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Risk Factors Detected
          </h4>
          <ul className="text-sm text-red-700 space-y-1">
            {riskFactors.map((factor, index) => (
              <li key={index} className="flex items-center gap-2">
                <div className="w-1 h-1 bg-red-500 rounded-full" />
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Positive Factors */}
      {positiveFactors.length > 0 && (
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h4 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Positive Indicators
          </h4>
          <ul className="text-sm text-green-700 space-y-1">
            {positiveFactors.map((factor, index) => (
              <li key={index} className="flex items-center gap-2">
                <div className="w-1 h-1 bg-green-500 rounded-full" />
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
          <Brain className="h-4 w-4" />
          Recommendations
        </h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li className="flex items-center gap-2">
            <div className="w-1 h-1 bg-blue-500 rounded-full" />
            Cross-reference with multiple reliable news sources
          </li>
          <li className="flex items-center gap-2">
            <div className="w-1 h-1 bg-blue-500 rounded-full" />
            Check the author's credentials and publication history
          </li>
          <li className="flex items-center gap-2">
            <div className="w-1 h-1 bg-blue-500 rounded-full" />
            Verify claims with fact-checking websites
          </li>
          <li className="flex items-center gap-2">
            <div className="w-1 h-1 bg-blue-500 rounded-full" />
            Look for original sources and citations
          </li>
        </ul>
      </div>
    </div>
  );
};

export default ResultsDisplay;