"""
ClimateAI - Model Training Module
SDG 13: Climate Action - Carbon Emission Forecasting

This module handles machine learning model training, evaluation, and optimization
for carbon emission prediction.

Author: PLP Academy Student = Nercia Motsepe
Assignment: Week 2 - AI for Sustainable Development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClimateEmissionPredictor:
    """
    A comprehensive machine learning class for climate emission prediction.
    
    This class implements:
    - Multiple ML algorithms comparison
    - Hyperparameter optimization
    - Model evaluation and validation
    - Feature importance analysis
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the predictor with random state for reproducibility.
        
        Args:
            random_state (int): Random seed for consistent results
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def initialize_models(self):
        """
        Initialize multiple ML models for comparison.
        """
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Linear Regression': LinearRegression(),
        }
        
        print(f"🤖 Initialized {len(self.models)} models for comparison")
        
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, X_train_scaled=None, X_test_scaled=None):
        """
        Train and evaluate all models.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            X_train_scaled, X_test_scaled: Scaled features for linear models
        """
        print("🚀 Training and evaluating models...")
        print("-" * 40)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for Linear Regression
            if name == 'Linear Regression' and X_train_scaled is not None:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"✅ {name} Results:")
            print(f"   • R² Score: {r2:.4f}")
            print(f"   • MAE: {mae:.4f}")
            print(f"   • RMSE: {rmse:.4f}")
            print(f"   • CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Identify best model
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"🎯 Best R² Score: {self.results[self.best_model_name]['r2']:.4f}")
        
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize hyperparameters for the best model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.best_model_name == 'Random Forest':
            print("🔧 Optimizing Random Forest hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            
            print(f"✅ Hyperparameter optimization completed!")
            print(f"🎯 Best Parameters: {grid_search.best_params_}")
            print(f"📈 Best CV Score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_params_
        else:
            print(f"ℹ️ No hyperparameter optimization for {self.best_model_name}")
            return None
    
    def analyze_feature_importance(self, feature_columns):
        """
        Analyze feature importance for tree-based models.
        
        Args:
            feature_columns: List of feature names
        """
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🎯 Feature Importance Analysis:")
            print("-" * 30)
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
            
            return self.feature_importance
        else:
            print(f"ℹ️ Feature importance not available for {self.best_model_name}")
            return None
    
    def generate_climate_insights(self):
        """
        Generate climate action insights based on model results.
        """
        insights = []
        
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(5)['feature'].tolist()
            
            if 'Energy_consumption' in top_features:
                insights.append("🔋 Energy transition is crucial for emission reduction")
            
            if 'Renewable_energy_pct' in top_features:
                insights.append("🌱 Renewable energy adoption shows strong impact")
            
            if 'Policy_score' in top_features or 'Policy_effectiveness' in top_features:
                insights.append("📋 Policy interventions demonstrate measurable effects")
            
            if 'GDP_per_capita' in top_features:
                insights.append("💰 Economic development patterns influence emissions")
            
            if 'Industrial_activity' in top_features:
                insights.append("🏭 Industrial efficiency improvements are key")
        
        return insights
    
    def create_scenario_analysis(self, sample_features, feature_columns):
        """
        Create policy scenario analysis for climate action planning.
        
        Args:
            sample_features: Sample country features for analysis
            feature_columns: List of feature names
            
        Returns:
            dict: Scenario results
        """
        print("🌍 Generating Climate Policy Scenarios...")
        
        base_features = sample_features.reshape(1, -1)
        baseline_emission = self.best_model.predict(base_features)[0]
        
        scenarios = {
            'Baseline (Current)': baseline_emission,
        }
        
        # Scenario 1: Increase renewable energy by 20%
        if 'Renewable_energy_pct' in feature_columns:
            renewable_scenario = base_features.copy()
            renewable_idx = feature_columns.index('Renewable_energy_pct')
            renewable_scenario[0, renewable_idx] = min(100, renewable_scenario[0, renewable_idx] * 1.2)
            scenarios['20% More Renewable Energy'] = self.best_model.predict(renewable_scenario)[0]
        
        # Scenario 2: Improve policy score by 30%
        if 'Policy_score' in feature_columns:
            policy_scenario = base_features.copy()
            policy_idx = feature_columns.index('Policy_score')
            policy_scenario[0, policy_idx] = min(100, policy_scenario[0, policy_idx] * 1.3)
            scenarios['30% Better Climate Policies'] = self.best_model.predict(policy_scenario)[0]
        
        # Scenario 3: Reduce energy consumption by 15%
        if 'Energy_consumption' in feature_columns:
            energy_scenario = base_features.copy()
            energy_idx = feature_columns.index('Energy_consumption')
            energy_scenario[0, energy_idx] = energy_scenario[0, energy_idx] * 0.85
            scenarios['15% Energy Efficiency'] = self.best_model.predict(energy_scenario)[0]
        
        # Scenario 4: Combined intervention
        combined_scenario = base_features.copy()
        if 'Renewable_energy_pct' in feature_columns:
            renewable_idx = feature_columns.index('Renewable_energy_pct')
            combined_scenario[0, renewable_idx] = min(100, combined_scenario[0, renewable_idx] * 1.2)
        if 'Policy_score' in feature_columns:
            policy_idx = feature_columns.index('Policy_score')
            combined_scenario[0, policy_idx] = min(100, combined_scenario[0, policy_idx] * 1.3)
        if 'Energy_consumption' in feature_columns:
            energy_idx = feature_columns.index('Energy_consumption')
            combined_scenario[0, energy_idx] = combined_scenario[0, energy_idx] * 0.85
        
        scenarios['Combined Climate Action'] = self.best_model.predict(combined_scenario)[0]
        
        # Calculate percentage changes
        scenario_analysis = {}
        for scenario, emission in scenarios.items():
            if scenario == 'Baseline (Current)':
                scenario_analysis[scenario] = {
                    'emission': emission,
                    'change_pct': 0.0
                }
            else:
                change_pct = ((baseline_emission - emission) / baseline_emission) * 100
                scenario_analysis[scenario] = {
                    'emission': emission,
                    'change_pct': change_pct
                }
        
        return scenario_analysis
    
    def visualize_results(self, y_test):
        """
        Create comprehensive visualizations of model results.
        
        Args:
            y_test: Test targets
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🤖 ClimateAI Model Performance - SDG 13 Climate Action', fontsize=16, fontweight='bold')
        
        # Model comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2'] for name in model_names]
        mae_scores = [self.results[name]['mae'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        axes[0, 1].bar(model_names, mae_scores, color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Mean Absolute Error Comparison')
        axes[0, 1].set_ylabel('MAE (Mt CO₂)')
        for i, v in enumerate(mae_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Prediction vs Actual for best model
        best_predictions = self.results[self.best_model_name]['predictions']
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6, color='blue')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'{self.best_model_name}: Predicted vs Actual')
        axes[1, 0].set_xlabel('Actual CO₂ Emissions')
        axes[1, 0].set_ylabel('Predicted CO₂ Emissions')
        
        # Feature importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'], color='green')
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_title('Top Feature Importance')
            axes[1, 1].set_xlabel('Importance Score')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and metadata.
        
        Args:
            model_dir: Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'climate_emission_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.best_model_name,
            'r2_score': self.results[self.best_model_name]['r2'],
            'mae': self.results[self.best_model_name]['mae'],
            'rmse': self.results[self.best_model_name]['rmse'],
            'cv_score': self.results[self.best_model_name]['cv_mean'],
            'training_date': datetime.now().isoformat(),
            'model_params': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {}
        }
        
        import json
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved successfully!")
        print(f"📁 Model file: {model_path}")
        print(f"📄 Metadata file: {metadata_path}")
        
        return model_path, metadata_path

def main():
    """
    Main function to demonstrate the model training pipeline.
    """
    print("🤖 ClimateAI Model Training Pipeline")
    print("=" * 50)
    
    # This would typically import from data_preprocessing.py
    from data_preprocessing import ClimateDataPreprocessor
    
    # Initialize components
    preprocessor = ClimateDataPreprocessor(random_state=42)
    predictor = ClimateEmissionPredictor(random_state=42)
    
    # Generate and preprocess data
    df = preprocessor.generate_climate_dataset()
    X, y, feature_columns = preprocessor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=df['Development_level']
    )
    
    # Scale features for linear models
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Initialize and train models
    predictor.initialize_models()
    predictor.train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled
    )
    
    # Optimize best model
    predictor.optimize_hyperparameters(X_train, y_train)
    
    # Analyze feature importance
    predictor.analyze_feature_importance(feature_columns)
    
    # Generate insights
    insights = predictor.generate_climate_insights()
    print(f"\n🌍 Climate Action Insights:")
    for insight in insights:
        print(f"   {insight}")
    
    # Create scenario analysis
    sample_features = X_test.iloc[0].values
    scenarios = predictor.create_scenario_analysis(sample_features, feature_columns)
    
    print(f"\n📊 Policy Scenario Analysis:")
    for scenario, data in scenarios.items():
        if data['change_pct'] == 0:
            print(f"   • {scenario}: {data['emission']:.2f} Mt CO₂")
        else:
            print(f"   • {scenario}: {data['emission']:.2f} Mt CO₂ ({data['change_pct']:+.1f}%)")
    
    # Visualize results
    predictor.visualize_results(y_test)
    
    # Save model
    predictor.save_model()
    
    print(f"\n🎯 Training completed successfully!")
    print(f"🏆 Best Model: {predictor.best_model_name}")
    print(f"📈 R² Score: {predictor.results[predictor.best_model_name]['r2']:.4f}")
    
    return predictor

if __name__ == "__main__":
    main()
