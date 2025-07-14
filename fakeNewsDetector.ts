/**
 * Fake news detection utilities and algorithms
 */

import { TextProcessor, TextStats, SentimentAnalysis } from './textProcessor';

export interface DetectionResult {
  label: 'Real' | 'Fake';
  confidence: number;
  probabilities: {
    fake: number;
    real: number;
  };
  factors: {
    linguistic: number;
    structural: number;
    emotional: number;
    credibility: number;
  };
  warnings: string[];
}

export interface ModelWeights {
  exclamation_weight: number;
  capital_weight: number;
  sentiment_weight: number;
  length_weight: number;
  punctuation_weight: number;
  emotional_weight: number;
  formality_weight: number;
}

export class FakeNewsDetector {
  private textProcessor: TextProcessor;
  private modelWeights: ModelWeights;

  constructor() {
    this.textProcessor = new TextProcessor();
    
    // Pre-trained weights (in reality, these would come from ML model)
    this.modelWeights = {
      exclamation_weight: -0.3,
      capital_weight: -0.4,
      sentiment_weight: -0.2,
      length_weight: 0.1,
      punctuation_weight: -0.2,
      emotional_weight: -0.5,
      formality_weight: 0.3
    };
  }

  async detectFakeNews(text: string): Promise<DetectionResult> {
    const features = this.textProcessor.extractFeatures(text);
    
    // Calculate component scores
    const linguisticScore = this.calculateLinguisticScore(features);
    const structuralScore = this.calculateStructuralScore(features);
    const emotionalScore = this.calculateEmotionalScore(features);
    const credibilityScore = this.calculateCredibilityScore(features);
    
    // Combine scores with weights
    const finalScore = this.combineScores({
      linguistic: linguisticScore,
      structural: structuralScore,
      emotional: emotionalScore,
      credibility: credibilityScore
    });
    
    // Convert to probabilities
    const fakeProbability = this.sigmoid(finalScore);
    const realProbability = 1 - fakeProbability;
    
    // Determine label and confidence
    const label = realProbability > fakeProbability ? 'Real' : 'Fake';
    const confidence = Math.max(realProbability, fakeProbability);
    
    // Generate warnings
    const warnings = this.generateWarnings(features, {
      linguistic: linguisticScore,
      structural: structuralScore,
      emotional: emotionalScore,
      credibility: credibilityScore
    });
    
    return {
      label,
      confidence,
      probabilities: {
        fake: fakeProbability,
        real: realProbability
      },
      factors: {
        linguistic: linguisticScore,
        structural: structuralScore,
        emotional: emotionalScore,
        credibility: credibilityScore
      },
      warnings
    };
  }

  private calculateLinguisticScore(features: Record<string, number>): number {
    let score = 0.5; // Start neutral
    
    // Exclamation marks (too many is suspicious)
    if (features.exclamationCount > 3) {
      score -= 0.2 * (features.exclamationCount / 10);
    }
    
    // Capital letters ratio
    if (features.capitalRatio > 0.05) {
      score -= 0.3 * (features.capitalRatio * 10);
    }
    
    // Punctuation ratio
    if (features.punctuationRatio > 0.1) {
      score -= 0.2 * (features.punctuationRatio * 5);
    }
    
    // Vocabulary diversity
    if (features.uniqueWordRatio > 0.5) {
      score += 0.1; // Good vocabulary diversity
    }
    
    return Math.max(0, Math.min(1, score));
  }

  private calculateStructuralScore(features: Record<string, number>): number {
    let score = 0.5;
    
    // Word count (very short articles are suspicious)
    if (features.wordCount < 100) {
      score -= 0.3;
    } else if (features.wordCount > 500) {
      score += 0.2; // Substantial articles are better
    }
    
    // Average sentence length
    if (features.avg_sentence_length > 30) {
      score -= 0.1; // Very long sentences might be confusing
    } else if (features.avg_sentence_length > 15) {
      score += 0.1; // Good sentence length
    }
    
    // Readability
    if (features.readability_score < 0.5) {
      score += 0.1; // More readable is better
    }
    
    return Math.max(0, Math.min(1, score));
  }

  private calculateEmotionalScore(features: Record<string, number>): number {
    let score = 0.5;
    
    // Sentiment extremes are suspicious
    if (Math.abs(features.sentiment_polarity) > 0.5) {
      score -= 0.2;
    }
    
    // High subjectivity is suspicious
    if (features.sentiment_subjectivity > 0.7) {
      score -= 0.3;
    }
    
    // Emotional intensity
    if (features.emotional_intensity > 0.3) {
      score -= 0.4;
    }
    
    return Math.max(0, Math.min(1, score));
  }

  private calculateCredibilityScore(features: Record<string, number>): number {
    let score = 0.5;
    
    // Formality score
    if (features.formality_score > 0) {
      score += 0.2; // More formal is better
    }
    
    // Use pre-calculated credibility indicators
    score = (score + features.credibility_indicators) / 2;
    
    return Math.max(0, Math.min(1, score));
  }

  private combineScores(scores: {
    linguistic: number;
    structural: number;
    emotional: number;
    credibility: number;
  }): number {
    // Weighted combination
    const weights = {
      linguistic: 0.3,
      structural: 0.2,
      emotional: 0.3,
      credibility: 0.2
    };
    
    return (
      scores.linguistic * weights.linguistic +
      scores.structural * weights.structural +
      scores.emotional * weights.emotional +
      scores.credibility * weights.credibility
    );
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x * 10)); // Scale for better separation
  }

  private generateWarnings(
    features: Record<string, number>,
    scores: { linguistic: number; structural: number; emotional: number; credibility: number }
  ): string[] {
    const warnings: string[] = [];
    
    if (features.exclamationCount > 5) {
      warnings.push('Excessive use of exclamation marks detected');
    }
    
    if (features.capitalRatio > 0.05) {
      warnings.push('High proportion of capital letters suggests sensationalism');
    }
    
    if (features.emotional_intensity > 0.3) {
      warnings.push('Highly emotional language may indicate bias');
    }
    
    if (features.wordCount < 100) {
      warnings.push('Very short article length reduces credibility');
    }
    
    if (features.sentiment_subjectivity > 0.7) {
      warnings.push('Highly subjective content detected');
    }
    
    if (Math.abs(features.sentiment_polarity) > 0.5) {
      warnings.push('Extreme sentiment polarity suggests bias');
    }
    
    if (scores.credibility < 0.3) {
      warnings.push('Low credibility indicators detected');
    }
    
    if (features.punctuation_ratio > 0.1) {
      warnings.push('Excessive punctuation may indicate sensationalism');
    }
    
    return warnings;
  }

  // Method to update model weights (for training)
  updateWeights(newWeights: Partial<ModelWeights>): void {
    this.modelWeights = { ...this.modelWeights, ...newWeights };
  }

  // Method to get current model performance metrics
  getModelInfo(): {
    version: string;
    features: string[];
    weights: ModelWeights;
  } {
    return {
      version: '1.0.0',
      features: [
        'word_count',
        'sentence_count',
        'avg_word_length',
        'exclamation_count',
        'question_count',
        'capital_ratio',
        'punctuation_ratio',
        'unique_word_ratio',
        'sentiment_polarity',
        'sentiment_subjectivity',
        'emotional_intensity',
        'avg_sentence_length',
        'readability_score',
        'formality_score',
        'credibility_indicators'
      ],
      weights: this.modelWeights
    };
  }
}

export default FakeNewsDetector;