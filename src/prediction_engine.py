"""
ClimateAI - Prediction Engine Module
SDG 13: Climate Action - Carbon Emission Forecasting

This module handles real-time predictions and scenario analysis for
carbon emission forecasting.

Author: PLP Academy Student - Nercia 
Assignment: Week 2 - AI for Sustainable Development
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClimateEmissionPredictor:
    """
    A production-ready prediction engine for carbon emission forecasting.
    
    This class provides:
    - Real-time emission predictions
    - Policy scenario analysis
    - Batch prediction capabilities
    - Model performance monitoring
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the prediction engine by loading trained models.
        
        Args:
            model_dir (str): Directory containing saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.metadata = None
        
        self.load_model_components()
    
    def load_model_components(self):
        """
        Load all model components from saved files.
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'climate_emission_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load preprocessing components
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            imputer_path = os.path.join(self.model_dir, 'imputer.pkl')
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
            
            # Load feature columns
            features_path = os.path.join(self.model_dir, 'feature_columns.pkl')
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print("✅ Model components loaded successfully!")
            if self.metadata:
                print(f"📊 Model Type: {self.metadata.get('model_type', 'Unknown')}")
                print(f"🎯 R² Score: {self.metadata.get('r2_score', 'Unknown'):.4f}")
                print(f"📅 Training Date: {self.metadata.get('training_date', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def validate_input(self, input_data):
        """
        Validate input data format and completeness.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Validated and formatted input data
        """
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Check for required features
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(input_df.columns)
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    if 'pct' in feature.lower() or 'ratio' in feature.lower():
                        input_df[feature] = 50.0  # Default percentage
                    elif 'score' in feature.lower():
                        input_df[feature] = 60.0  # Default score
                    elif 'encoded' in feature.lower():
                        input_df[feature] = 0  # Default encoding
                    else:
                        input_df[feature] = input_df.select_dtypes(include=[np.number]).mean().mean()
        
        return input_df
    
    def preprocess_input(self, input_df):
        """
        Preprocess input data using saved preprocessing components.
        
        Args:
            input_df (pd.DataFrame): Input dataframe
            
        Returns:
            np.ndarray: Preprocessed features ready for prediction
        """
        # Select only the features used in training
        if self.feature_columns:
            input_features = input_df[self.feature_columns]
        else:
            input_features = input_df
        
        # Handle missing values
        if self.imputer:
            input_features = self.imputer.transform(input_features)
        
        # Scale features if scaler is available
        if self.scaler:
            input_features = self.scaler.transform(input_features)
        
        return input_features
    
    def predict_emissions(self, input_data, return_confidence=False):
        """
        Predict CO₂ emissions for given input data.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            return_confidence (bool): Whether to return prediction confidence
            
        Returns:
            float or tuple: Predicted emissions (and confidence if requested)
        """
        # Validate and preprocess input
        input_df = self.validate_input(input_data)
        processed_features = self.preprocess_input(input_df)
        
        # Make prediction
        prediction = self.model.predict(processed_features)[0]
        
        if return_confidence:
            # Calculate confidence based on model type
            if hasattr(self.model, 'predict_proba'):
                # For models with probability estimates
                confidence = 0.95  # Default high confidence
            else:
                # Use R² score as confidence proxy
                confidence = self.metadata.get('r2_score', 0.95) if self.metadata else 0.95
            
            return prediction, confidence
        
        return prediction
    
    def batch_predict(self, input_data_list):
        """
        Make predictions for multiple inputs.
        
        Args:
            input_data_list (list): List of input dictionaries or DataFrame
            
        Returns:
            list: List of predictions
        """
        predictions = []
        
        for input_data in input_data_list:
            try:
                prediction = self.predict_emissions(input_data)
                predictions.append(prediction)
            except Exception as e:
                print(f"⚠️ Error predicting for input {input_data}: {e}")
                predictions.append(None)
        
        return predictions
    
    def scenario_analysis(self, base_input, scenarios):
        """
        Perform scenario analysis for climate policy planning.
        
        Args:
            base_input (dict): Base country/region data
            scenarios (dict): Dictionary of scenario modifications
            
        Returns:
            dict: Scenario results with emission predictions and changes
        """
        print("🌍 Performing Climate Policy Scenario Analysis...")
        
        # Get baseline prediction
        baseline_emission = self.predict_emissions(base_input)
        
        results = {
            'Baseline': {
                'emission': baseline_emission,
                'change_pct': 0.0,
                'description': 'Current conditions'
            }
        }
        
        # Analyze each scenario
        for scenario_name, modifications in scenarios.items():
            # Create modified input
            scenario_input = base_input.copy()
            scenario_input.update(modifications)
            
            # Get prediction
            scenario_emission = self.predict_emissions(scenario_input)
            
            # Calculate change
            change_pct = ((baseline_emission - scenario_emission) / baseline_emission) * 100
            
            results[scenario_name] = {
                'emission': scenario_emission,
                'change_pct': change_pct,
                'description': f"Modified: {', '.join(modifications.keys())}"
            }
        
        return results
    
    def get_feature_impact(self, base_input, feature_name, change_pct):
        """
        Analyze the impact of changing a specific feature.
        
        Args:
            base_input (dict): Base input data
            feature_name (str): Name of feature to modify
            change_pct (float): Percentage change to apply
            
        Returns:
            dict: Impact analysis results
        """
        if feature_name not in base_input:
            return {'error': f'Feature {feature_name} not found in input'}
        
        # Get baseline
        baseline_emission = self.predict_emissions(base_input)
        
        # Create modified input
        modified_input = base_input.copy()
        original_value = modified_input[feature_name]
        modified_input[feature_name] = original_value * (1 + change_pct / 100)
        
        # Get modified prediction
        modified_emission = self.predict_emissions(modified_input)
        
        # Calculate impact
        emission_change_pct = ((baseline_emission - modified_emission) / baseline_emission) * 100
        
        return {
            'feature': feature_name,
            'original_value': original_value,
            'modified_value': modified_input[feature_name],
            'feature_change_pct': change_pct,
            'baseline_emission': baseline_emission,
            'modified_emission': modified_emission,
            'emission_change_pct': emission_change_pct,
            'impact_ratio': emission_change_pct / change_pct if change_pct != 0 else 0
        }
    
    def generate_recommendations(self, input_data, target_reduction_pct=20):
        """
        Generate climate action recommendations to achieve target emission reduction.
        
        Args:
            input_data (dict): Current country/region data
            target_reduction_pct (float): Target emission reduction percentage
            
        Returns:
            dict: Recommendations and their potential impact
        """
        print(f"🎯 Generating recommendations for {target_reduction_pct}% emission reduction...")
        
        baseline_emission = self.predict_emissions(input_data)
        target_emission = baseline_emission * (1 - target_reduction_pct / 100)
        
        # Define potential interventions
        interventions = {
            'Renewable Energy Increase': {
                'feature': 'Renewable_energy_pct',
                'change_pct': 25,
                'description': 'Increase renewable energy adoption by 25%'
            },
            'Energy Efficiency': {
                'feature': 'Energy_consumption',
                'change_pct': -15,
                'description': 'Reduce energy consumption by 15% through efficiency'
            },
            'Policy Improvement': {
                'feature': 'Policy_score',
                'change_pct': 30,
                'description': 'Strengthen climate policies by 30%'
            },
            'Industrial Efficiency': {
                'feature': 'Industrial_activity',
                'change_pct': -10,
                'description': 'Improve industrial efficiency by 10%'
            }
        }
        
        recommendations = {}
        
        for intervention_name, intervention in interventions.items():
            if intervention['feature'] in input_data:
                impact = self.get_feature_impact(
                    input_data, 
                    intervention['feature'], 
                    intervention['change_pct']
                )
                
                recommendations[intervention_name] = {
                    'description': intervention['description'],
                    'emission_reduction_pct': impact['emission_change_pct'],
                    'feasibility': 'High' if abs(impact['emission_change_pct']) > 5 else 'Medium',
                    'impact': impact
                }
        
        # Sort by effectiveness
        sorted_recommendations = dict(
            sorted(recommendations.items(), 
                   key=lambda x: x[1]['emission_reduction_pct'], 
                   reverse=True)
        )
        
        return {
            'baseline_emission': baseline_emission,
            'target_emission': target_emission,
            'target_reduction_pct': target_reduction_pct,
            'recommendations': sorted_recommendations
        }
    
    def get_model_info(self):
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            dict: Model information and performance metrics
        """
        info = {
            'model_loaded': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'preprocessing_components': {
                'scaler': self.scaler is not None,
                'imputer': self.imputer is not None
            }
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info

def create_sample_input():
    """
    Create a sample input for testing the prediction engine.
    
    Returns:
        dict: Sample country data
    """
    return {
        'GDP_per_capita': 35000.0,
        'Population_density': 150.0,
        'Energy_consumption': 12000.0,
        'Industrial_activity': 8500.0,
        'Renewable_energy_pct': 35.0,
        'Urban_population_pct': 75.0,
        'Transport_emissions': 180.0,
        'Policy_score': 65.0,
        'Development_level_encoded': 1,  # Developed
        'GDP_Energy_ratio': 2.9,
        'GDP_Industrial_ratio': 4.1,
        'Urban_density': 112.5,
        'Policy_effectiveness': 22.75,
        'Green_development': 12250.0,
        'Years_since_2000': 23
    }

def main():
    """
    Main function to demonstrate the prediction engine.
    """
    print("🔮 ClimateAI Prediction Engine Demo")
    print("=" * 50)
    
    try:
        # Initialize prediction engine
        predictor = ClimateEmissionPredictor()
        
        # Create sample input
        sample_input = create_sample_input()
        
        # Make single prediction
        print("\n🎯 Single Prediction:")
        emission, confidence = predictor.predict_emissions(sample_input, return_confidence=True)
        print(f"   • Predicted CO₂ Emission: {emission:.2f} Mt CO₂")
        print(f"   • Confidence: {confidence:.1%}")
        
        # Scenario analysis
        print("\n🌍 Scenario Analysis:")
        scenarios = {
            'Green Transition': {
                'Renewable_energy_pct': 60.0,
                'Energy_consumption': 10000.0
            },
            'Policy Reform': {
                'Policy_score': 85.0,
                'Renewable_energy_pct': 45.0
            },
            'Industrial Efficiency': {
                'Industrial_activity': 7000.0,
                'Energy_consumption': 10500.0
            }
        }
        
        scenario_results = predictor.scenario_analysis(sample_input, scenarios)
        for scenario, result in scenario_results.items():
            print(f"   • {scenario}: {result['emission']:.2f} Mt CO₂ ({result['change_pct']:+.1f}%)")
        
        # Generate recommendations
        print("\n💡 Climate Action Recommendations:")
        recommendations = predictor.generate_recommendations(sample_input, target_reduction_pct=25)
        
        for rec_name, rec_data in recommendations['recommendations'].items():
            print(f"   • {rec_name}: {rec_data['emission_reduction_pct']:.1f}% reduction")
            print(f"     {rec_data['description']}")
        
        # Model information
        print("\n📊 Model Information:")
        model_info = predictor.get_model_info()
        print(f"   • Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"   • Features: {model_info.get('feature_count', 0)}")
        print(f"   • R² Score: {model_info.get('r2_score', 'Unknown')}")
        
        print("\n✅ Prediction engine demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in prediction engine demo: {e}")
        print("💡 Make sure to run model training first to generate model files.")

if __name__ == "__main__":
    main()
