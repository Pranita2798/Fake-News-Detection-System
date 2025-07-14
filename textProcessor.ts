/**
 * Text processing utilities for fake news detection
 */

export interface TextStats {
  wordCount: number;
  sentenceCount: number;
  avgWordLength: number;
  exclamationCount: number;
  questionCount: number;
  capitalRatio: number;
  punctuationRatio: number;
  uniqueWordRatio: number;
}

export interface SentimentAnalysis {
  polarity: number; // -1 to 1
  subjectivity: number; // 0 to 1
  emotionalIntensity: number;
}

export class TextProcessor {
  private stopWords = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your'
  ]);

  private positiveWords = new Set([
    'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'incredible',
    'outstanding', 'perfect', 'wonderful', 'brilliant', 'good', 'best',
    'love', 'beautiful', 'happy', 'joy', 'success', 'win', 'triumph'
  ]);

  private negativeWords = new Set([
    'awful', 'terrible', 'horrible', 'disgusting', 'hate', 'worst',
    'fail', 'disaster', 'crisis', 'danger', 'threat', 'problem',
    'bad', 'wrong', 'evil', 'corrupt', 'scandal', 'shocking'
  ]);

  private emotionalWords = new Set([
    'shocking', 'unbelievable', 'incredible', 'amazing', 'devastating',
    'outrageous', 'scandal', 'exposed', 'revealed', 'secret', 'hidden',
    'urgent', 'breaking', 'exclusive', 'explosive', 'bombshell'
  ]);

  cleanText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s.,!?;:'"()-]/g, '') // Keep basic punctuation
      .replace(/\s+/g, ' ')
      .trim();
  }

  extractTextStats(text: string): TextStats {
    const cleaned = this.cleanText(text);
    const words = cleaned.split(/\s+/).filter(word => word.length > 0);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    const uniqueWords = new Set(words.filter(word => !this.stopWords.has(word)));
    const totalChars = text.length;
    
    return {
      wordCount: words.length,
      sentenceCount: sentences.length,
      avgWordLength: words.reduce((sum, word) => sum + word.length, 0) / words.length || 0,
      exclamationCount: (text.match(/!/g) || []).length,
      questionCount: (text.match(/\?/g) || []).length,
      capitalRatio: (text.match(/[A-Z]/g) || []).length / totalChars,
      punctuationRatio: (text.match(/[.,!?;:'"()-]/g) || []).length / totalChars,
      uniqueWordRatio: uniqueWords.size / words.length || 0
    };
  }

  analyzeSentiment(text: string): SentimentAnalysis {
    const words = this.cleanText(text).split(/\s+/);
    
    let positiveScore = 0;
    let negativeScore = 0;
    let emotionalScore = 0;
    let subjectiveWords = 0;
    
    words.forEach(word => {
      if (this.positiveWords.has(word)) {
        positiveScore++;
        subjectiveWords++;
      }
      if (this.negativeWords.has(word)) {
        negativeScore++;
        subjectiveWords++;
      }
      if (this.emotionalWords.has(word)) {
        emotionalScore++;
        subjectiveWords++;
      }
    });

    const totalWords = words.length;
    const polarity = (positiveScore - negativeScore) / totalWords;
    const subjectivity = subjectiveWords / totalWords;
    const emotionalIntensity = emotionalScore / totalWords;

    return {
      polarity: Math.max(-1, Math.min(1, polarity)),
      subjectivity: Math.max(0, Math.min(1, subjectivity)),
      emotionalIntensity: Math.max(0, Math.min(1, emotionalIntensity))
    };
  }

  extractFeatures(text: string): Record<string, number> {
    const stats = this.extractTextStats(text);
    const sentiment = this.analyzeSentiment(text);
    
    return {
      ...stats,
      sentiment_polarity: sentiment.polarity,
      sentiment_subjectivity: sentiment.subjectivity,
      emotional_intensity: sentiment.emotionalIntensity,
      
      // Additional features
      avg_sentence_length: stats.wordCount / stats.sentenceCount || 0,
      readability_score: this.calculateReadabilityScore(stats),
      formality_score: this.calculateFormalityScore(text),
      credibility_indicators: this.calculateCredibilityIndicators(text, stats)
    };
  }

  private calculateReadabilityScore(stats: TextStats): number {
    // Simple readability score based on average word and sentence length
    const avgWordsPerSentence = stats.wordCount / stats.sentenceCount || 0;
    const complexity = (stats.avgWordLength * 0.4) + (avgWordsPerSentence * 0.6);
    
    // Normalize to 0-1 scale (lower is more readable)
    return Math.min(1, complexity / 20);
  }

  private calculateFormalityScore(text: string): number {
    const formalWords = ['however', 'therefore', 'consequently', 'furthermore', 'moreover'];
    const informalWords = ['gonna', 'wanna', 'gotta', 'kinda', 'sorta'];
    
    const words = this.cleanText(text).split(/\s+/);
    
    let formalCount = 0;
    let informalCount = 0;
    
    words.forEach(word => {
      if (formalWords.includes(word)) formalCount++;
      if (informalWords.includes(word)) informalCount++;
    });
    
    // Return score from -1 (informal) to 1 (formal)
    return (formalCount - informalCount) / words.length;
  }

  private calculateCredibilityIndicators(text: string, stats: TextStats): number {
    let score = 0.5; // Start with neutral
    
    // Positive indicators
    if (stats.exclamationCount <= 2) score += 0.1;
    if (stats.capitalRatio <= 0.03) score += 0.1;
    if (stats.wordCount >= 200) score += 0.1;
    if (stats.questionCount <= 2) score += 0.05;
    
    // Negative indicators
    if (stats.exclamationCount > 5) score -= 0.2;
    if (stats.capitalRatio > 0.05) score -= 0.2;
    if (stats.wordCount < 100) score -= 0.1;
    if (text.includes('BREAKING:')) score -= 0.1;
    if (text.includes('SHOCKING')) score -= 0.1;
    
    return Math.max(0, Math.min(1, score));
  }

  // Feature extraction for machine learning
  vectorizeText(text: string): number[] {
    const features = this.extractFeatures(text);
    
    // Return feature vector in consistent order
    return [
      features.wordCount / 1000, // Normalize
      features.sentenceCount / 50,
      features.avgWordLength / 10,
      features.exclamationCount / 10,
      features.questionCount / 10,
      features.capitalRatio,
      features.punctuationRatio,
      features.uniqueWordRatio,
      features.sentiment_polarity,
      features.sentiment_subjectivity,
      features.emotional_intensity,
      features.avg_sentence_length / 20,
      features.readability_score,
      features.formality_score,
      features.credibility_indicators
    ];
  }
}

export default TextProcessor;