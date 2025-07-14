#!/usr/bin/env python3
"""
Model Trainer for Fake News Detection System
Trains machine learning models for fake news classification.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from nlp_preprocessor import NLPPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeNewsModelTrainer:
    def __init__(self):
        self.preprocessor = NLPPreprocessor()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        self.best_model = None
        self.best_score = 0
        self.feature_names = []
        
    def load_data(self, csv_path):
        """Load and prepare training data."""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_columns = ['content', 'label']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Remove rows with missing content
        df = df.dropna(subset=['content'])
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training."""
        logger.info("Preparing features...")
        
        # Use preprocessor to extract features
        X, feature_names = self.preprocessor.process_dataset(df)
        y = df['label'].values
        
        self.feature_names = feature_names
        
        return X, y
    
    def train_single_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """Train and evaluate a single model."""
        logger.info(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return model, metrics
    
    def hyperparameter_tuning(self, model_name, model, X_train, y_train):
        """Perform hyperparameter tuning."""
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def train_all_models(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True):
        """Train all models and compare performance."""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Hyperparameter tuning
                if tune_hyperparameters:
                    model = self.hyperparameter_tuning(model_name, model, X_train, y_train)
                
                # Train and evaluate
                trained_model, metrics = self.train_single_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                
                results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics
                }
                
                # Update best model
                if metrics['f1_score'] > self.best_score:
                    self.best_score = metrics['f1_score']
                    self.best_model = trained_model
                    logger.info(f"New best model: {model_name} (F1: {self.best_score:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation."""
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def feature_importance_analysis(self, model, model_name="Model"):
        """Analyze feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Show top 20 features
            print(f"\n{model_name} - Top 20 Feature Importances:")
            for i in range(min(20, len(indices))):
                feature_idx = indices[i]
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"Feature_{feature_idx}"
                print(f"{i+1}. {feature_name}: {importances[feature_idx]:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            plt.title(f'{model_name} Feature Importance (Top 20)')
            top_features = [self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}" 
                          for i in indices[:20]]
            plt.barh(range(20), importances[indices[:20]])
            plt.yticks(range(20), top_features)
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
    
    def save_model(self, model, model_name, save_path):
        """Save trained model."""
        model_data = {
            'model': model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'model_name': model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def cross_validation(self, model, X, y, cv=5):
        """Perform cross-validation."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {scores}")
        logger.info(f"Mean F1 score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores

def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = FakeNewsModelTrainer()
    
    # Load data (you'll need to provide actual dataset)
    # df = trainer.load_data('data/fake_news_dataset.csv')
    
    # Create sample data for demonstration
    sample_data = {
        'content': [
            "This is a well-researched article about climate change with proper citations.",
            "SHOCKING!!! Scientists HATE this one weird trick! Climate change is FAKE NEWS!!!",
            "A balanced report on economic indicators shows steady growth.",
            "You won't BELIEVE what the government is hiding from you! EXPOSED!!!",
            "The latest medical research indicates promising results for new treatment.",
            "DOCTORS HATE HIM! This man cured cancer with this ONE SIMPLE TRICK!",
            "Analysis of market trends suggests cautious optimism for investors.",
            "BREAKING: Celebrity scandal rocks the world! You won't believe what happened next!",
            "University study reveals new insights into renewable energy efficiency.",
            "URGENT: This message will change your life forever! Click now or regret it!"
        ] * 100,  # Repeat to have enough samples
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100  # 1 for real, 0 for fake
    }
    
    df = pd.DataFrame(sample_data)
    
    # Prepare features
    X, y = trainer.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\nModel Performance Summary:")
    print("-" * 60)
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:20} | F1: {metrics['f1_score']:.4f} | "
              f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc_score']:.4f}")
    
    # Evaluate best model
    if trainer.best_model:
        print(f"\nBest model performance:")
        trainer.evaluate_model(trainer.best_model, X_test, y_test, "Best Model")
        trainer.feature_importance_analysis(trainer.best_model, "Best Model")
        
        # Save best model
        trainer.save_model(trainer.best_model, "best_model", "models/best_fake_news_model.pkl")

if __name__ == "__main__":
    main()