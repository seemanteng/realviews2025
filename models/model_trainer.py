"""
Model training pipeline for RealViews
Supports retraining with custom datasets and model improvement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import json
import os
from pathlib import Path
import logging
from datetime import datetime

from utils.data_processing import ReviewProcessor
from utils.data_loader import ReviewDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training and evaluation system"""
    
    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models/saved_models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = ReviewProcessor()
        self.vectorizers = {}
        self.models = {}
        self.training_history = []
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            }
        }
    
    def load_training_data(self, use_existing: bool = True) -> Dict[str, pd.DataFrame]:
        """Load training data from processed files or create new splits"""
        
        if use_existing and (self.data_dir / "train_data.csv").exists():
            logger.info("Loading existing processed data...")
            splits = {}
            
            for split_name in ['train', 'validation', 'test']:
                file_path = self.data_dir / f"{split_name}_data.csv"
                if file_path.exists():
                    splits[split_name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {split_name}: {len(splits[split_name])} samples")
            
            return splits
        
        else:
            logger.info("Creating new data splits from raw data...")
            loader = ReviewDataLoader()
            
            # Load all available data
            csv_datasets = loader.load_csv_files()
            json_datasets = loader.load_json_files()
            all_datasets = csv_datasets + json_datasets
            
            if not all_datasets:
                logger.warning("No raw data found. Creating example datasets...")
                loader.load_example_datasets()
                csv_datasets = loader.load_csv_files()
                all_datasets = csv_datasets
            
            # Standardize and prepare
            standardized_datasets = []
            for df in all_datasets:
                df_std = loader.standardize_columns(df)
                df_std = loader.normalize_violation_labels(df_std)
                standardized_datasets.append(df_std)
            
            # Create splits
            splits = loader.prepare_training_data(standardized_datasets)
            loader.save_processed_data(splits)
            
            return splits
    
    def extract_features(self, texts: List[str], fit_vectorizer: bool = False) -> np.ndarray:
        """Extract comprehensive features from text data"""
        
        # TF-IDF features
        if fit_vectorizer or 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.vectorizers['tfidf'].fit_transform(texts)
        else:
            tfidf_features = self.vectorizers['tfidf'].transform(texts)
        
        # Manual features from ReviewProcessor
        manual_features = []
        for text in texts:
            features = self.processor.extract_features(text)
            feature_vector = [
                features.get('word_count', 0),
                features.get('sentence_count', 0),
                features.get('avg_word_length', 0),
                features.get('uppercase_ratio', 0),
                features.get('punctuation_ratio', 0),
                features.get('digit_ratio', 0),
                features.get('promotional_keywords', 0),
                features.get('fake_indicators', 0),
                features.get('irrelevant_indicators', 0),
                features.get('specific_details', 0),
                features.get('personal_experience', 0),
                features.get('sentiment_compound', 0) if 'sentiment_compound' in features else features.get('sentiment_polarity', 0),
                features.get('unique_word_ratio', 0),
                features.get('exclamation_count', 0),
                features.get('question_count', 0)
            ]
            manual_features.append(feature_vector)
        
        manual_features = np.array(manual_features)
        
        # Combine TF-IDF and manual features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([tfidf_features, csr_matrix(manual_features)])
        
        return combined_features
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          use_grid_search: bool = True) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization"""
        
        logger.info(f"Training {model_name}...")
        
        config = self.model_configs[model_name]
        base_model = config['model']
        
        if use_grid_search:
            logger.info(f"  Running grid search for {model_name}...")
            grid_search = GridSearchCV(
                base_model,
                config['param_grid'],
                cv=3,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"  Best params: {grid_search.best_params_}")
        else:
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_pred_proba = best_model.predict_proba(X_val) if hasattr(best_model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_val, val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='macro')
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='f1_macro')
        
        results = {
            'model': best_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'validation_predictions': val_pred,
            'validation_probabilities': val_pred_proba
        }
        
        logger.info(f"  {model_name} results: F1={f1:.3f}, Accuracy={accuracy:.3f}")
        
        return results
    
    def train_ensemble(self,
                      individual_results: Dict[str, Dict],
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray) -> Dict[str, Any]:
        """Create and train an ensemble of the best models"""
        
        logger.info("Training ensemble model...")
        
        # Select top 3 models based on F1 score
        sorted_models = sorted(individual_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        top_models = sorted_models[:3]
        
        ensemble_estimators = [(name, results['model']) for name, results in top_models]
        
        ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        val_pred = ensemble.predict(X_val)
        val_pred_proba = ensemble.predict_proba(X_val)
        
        accuracy = accuracy_score(y_val, val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='macro')
        
        results = {
            'model': ensemble,
            'component_models': [name for name, _ in top_models],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'validation_predictions': val_pred,
            'validation_probabilities': val_pred_proba
        }
        
        logger.info(f"  Ensemble F1={f1:.3f}, Accuracy={accuracy:.3f}")
        
        return results
    
    def train_violation_classifiers(self, 
                                  train_data: pd.DataFrame,
                                  val_data: pd.DataFrame,
                                  violation_types: List[str] = None) -> Dict[str, Dict]:
        """Train separate classifiers for each violation type"""
        
        if violation_types is None:
            violation_types = ['advertisement', 'fake', 'irrelevant']
        
        violation_classifiers = {}
        
        # Extract features once for all violation types
        X_train = self.extract_features(train_data['text'].tolist(), fit_vectorizer=True)
        X_val = self.extract_features(val_data['text'].tolist(), fit_vectorizer=False)
        
        for violation_type in violation_types:
            logger.info(f"\nTraining classifiers for {violation_type} violation...")
            
            # Create binary labels
            y_train = train_data.get(f'is_{violation_type}', 
                                   (train_data['violation_type'] == violation_type).astype(int)).values
            y_val = val_data.get(f'is_{violation_type}',
                               (val_data['violation_type'] == violation_type).astype(int)).values
            
            # Skip if no positive examples
            if y_train.sum() == 0:
                logger.warning(f"  No {violation_type} examples found in training data. Skipping.")
                continue
            
            # Train individual models
            individual_results = {}
            for model_name in ['logistic_regression', 'random_forest', 'naive_bayes']:
                try:
                    results = self.train_single_model(
                        model_name, X_train, y_train, X_val, y_val, use_grid_search=True
                    )
                    individual_results[model_name] = results
                except Exception as e:
                    logger.error(f"  Failed to train {model_name}: {e}")
            
            # Train ensemble if we have multiple models
            if len(individual_results) > 1:
                try:
                    ensemble_results = self.train_ensemble(
                        individual_results, X_train, y_train, X_val, y_val
                    )
                    individual_results['ensemble'] = ensemble_results
                except Exception as e:
                    logger.error(f"  Failed to train ensemble: {e}")
            
            violation_classifiers[violation_type] = individual_results
        
        return violation_classifiers
    
    def train_quality_predictor(self,
                              train_data: pd.DataFrame,
                              val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train a model to predict review quality scores"""
        
        logger.info("\nTraining quality prediction model...")
        
        # Calculate quality scores for training data
        quality_scores = []
        for text in train_data['text']:
            features = self.processor.extract_features(text)
            quality_score = self.processor.calculate_quality_score(features, text)
            quality_scores.append(quality_score)
        
        train_data = train_data.copy()
        train_data['quality_score'] = quality_scores
        
        # Same for validation
        val_quality_scores = []
        for text in val_data['text']:
            features = self.processor.extract_features(text)
            quality_score = self.processor.calculate_quality_score(features, text)
            val_quality_scores.append(quality_score)
        
        val_data = val_data.copy()
        val_data['quality_score'] = val_quality_scores
        
        # Extract features
        X_train = self.extract_features(train_data['text'].tolist(), fit_vectorizer=True)
        X_val = self.extract_features(val_data['text'].tolist(), fit_vectorizer=False)
        
        y_train = np.array(quality_scores)
        y_val = np.array(val_quality_scores)
        
        # Train regression model
        from sklearn.ensemble import RandomForestRegressor
        
        quality_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        quality_model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = quality_model.predict(X_val)
        mse = mean_squared_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        logger.info(f"  Quality model: R2={r2:.3f}, MSE={mse:.3f}")
        
        return {
            'model': quality_model,
            'r2_score': r2,
            'mse': mse,
            'validation_predictions': val_pred,
            'validation_actual': y_val
        }
    
    def evaluate_on_test_set(self,
                           trained_models: Dict[str, Any],
                           test_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive evaluation on test set"""
        
        logger.info("\nEvaluating models on test set...")
        
        evaluation_results = {}
        
        # Extract test features
        X_test = self.extract_features(test_data['text'].tolist(), fit_vectorizer=False)
        
        # Evaluate violation classifiers
        for violation_type, models in trained_models.get('violation_classifiers', {}).items():
            if 'ensemble' in models:
                best_model = models['ensemble']['model']
                model_name = 'ensemble'
            else:
                # Find best individual model
                best_name = max(models.keys(), key=lambda k: models[k]['f1_score'])
                best_model = models[best_name]['model']
                model_name = best_name
            
            # Test labels
            y_test = test_data.get(f'is_{violation_type}',
                                 (test_data['violation_type'] == violation_type).astype(int)).values
            
            # Predictions
            test_pred = best_model.predict(X_test)
            test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
            
            evaluation_results[f'{violation_type}_classifier'] = {
                'model_used': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': test_pred.tolist(),
                'probabilities': test_proba.tolist() if test_proba is not None else None
            }
            
            logger.info(f"  {violation_type}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        # Evaluate quality predictor
        if 'quality_predictor' in trained_models:
            quality_model = trained_models['quality_predictor']['model']
            
            # Calculate actual quality scores
            actual_quality = []
            for text in test_data['text']:
                features = self.processor.extract_features(text)
                quality_score = self.processor.calculate_quality_score(features, text)
                actual_quality.append(quality_score)
            
            pred_quality = quality_model.predict(X_test)
            
            mae = mean_absolute_error(actual_quality, pred_quality)
            mse = mean_squared_error(actual_quality, pred_quality)
            r2 = r2_score(actual_quality, pred_quality)
            
            evaluation_results['quality_predictor'] = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2,
                'predictions': pred_quality.tolist(),
                'actual': actual_quality
            }
            
            logger.info(f"  Quality prediction: R2={r2:.3f}, MAE={mae:.3f}")
        
        return evaluation_results
    
    def save_models(self, trained_models: Dict[str, Any], model_version: str = None) -> str:
        """Save trained models and metadata"""
        
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_dir = self.models_dir / f"v_{model_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {version_dir}")
        
        # Save individual models
        saved_files = {}
        
        # Violation classifiers
        for violation_type, models in trained_models.get('violation_classifiers', {}).items():
            for model_name, model_info in models.items():
                filename = f"{violation_type}_{model_name}_model.pkl"
                model_path = version_dir / filename
                joblib.dump(model_info['model'], model_path)
                saved_files[f"{violation_type}_{model_name}"] = str(model_path)
        
        # Quality predictor
        if 'quality_predictor' in trained_models:
            quality_path = version_dir / "quality_predictor.pkl"
            joblib.dump(trained_models['quality_predictor']['model'], quality_path)
            saved_files['quality_predictor'] = str(quality_path)
        
        # Save vectorizers
        vectorizers_path = version_dir / "vectorizers.pkl"
        joblib.dump(self.vectorizers, vectorizers_path)
        saved_files['vectorizers'] = str(vectorizers_path)
        
        # Save metadata
        metadata = {
            'version': model_version,
            'created_at': datetime.now().isoformat(),
            'model_files': saved_files,
            'training_results': self._serialize_results(trained_models),
            'feature_info': {
                'tfidf_vocab_size': len(self.vectorizers.get('tfidf', {}).vocabulary_) if 'tfidf' in self.vectorizers else 0,
                'manual_features_count': 15  # Number of manual features
            }
        }
        
        metadata_path = version_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  Saved {len(saved_files)} model files")
        logger.info(f"  Metadata saved to {metadata_path}")
        
        return model_version
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Convert results to JSON-serializable format"""
        serialized = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                if 'model' in value:
                    # Remove model object, keep metrics
                    serialized[key] = {k: v for k, v in value.items() 
                                     if k != 'model' and not isinstance(v, np.ndarray)}
                else:
                    serialized[key] = self._serialize_results(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def full_training_pipeline(self, 
                             use_existing_data: bool = True,
                             save_models: bool = True) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        logger.info("🚀 Starting full training pipeline...")
        
        # Load data
        splits = self.load_training_data(use_existing_data)
        
        if not splits:
            logger.error("No training data available!")
            return {}
        
        train_data = splits['train']
        val_data = splits['validation']
        test_data = splits.get('test')
        
        logger.info(f"Training with {len(train_data)} samples, validating with {len(val_data)}")
        
        # Train models
        results = {}
        
        # 1. Train violation classifiers
        violation_classifiers = self.train_violation_classifiers(train_data, val_data)
        results['violation_classifiers'] = violation_classifiers
        
        # 2. Train quality predictor
        quality_predictor = self.train_quality_predictor(train_data, val_data)
        results['quality_predictor'] = quality_predictor
        
        # 3. Evaluate on test set if available
        if test_data is not None and len(test_data) > 0:
            test_results = self.evaluate_on_test_set(results, test_data)
            results['test_evaluation'] = test_results
        
        # 4. Save models
        if save_models:
            model_version = self.save_models(results)
            results['model_version'] = model_version
        
        logger.info("✅ Training pipeline completed successfully!")
        
        return results

def main():
    """Example usage of the model trainer"""
    trainer = ModelTrainer()
    
    # Run full training pipeline
    results = trainer.full_training_pipeline(use_existing_data=True, save_models=True)
    
    if results:
        print("\n📊 Training Summary:")
        
        # Violation classifiers summary
        for violation_type, models in results.get('violation_classifiers', {}).items():
            best_f1 = max(model['f1_score'] for model in models.values())
            print(f"  {violation_type.title()}: Best F1 = {best_f1:.3f}")
        
        # Quality predictor summary
        if 'quality_predictor' in results:
            r2 = results['quality_predictor']['r2_score']
            print(f"  Quality Prediction: R2 = {r2:.3f}")
        
        # Test results summary
        if 'test_evaluation' in results:
            print("\n📋 Test Set Results:")
            for classifier_name, metrics in results['test_evaluation'].items():
                if classifier_name.endswith('_classifier'):
                    print(f"  {classifier_name}: F1 = {metrics['f1_score']:.3f}")
        
        if 'model_version' in results:
            print(f"\n💾 Models saved as version: {results['model_version']}")
    
    print("\n🎯 Training complete! Use the new models in your RealViews system.")

if __name__ == "__main__":
    main()