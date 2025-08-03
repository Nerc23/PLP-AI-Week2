"""
ClimateAI - Data Preprocessing Module
SDG 13: Climate Action - Carbon Emission Forecasting

This module handles data collection, cleaning, and preprocessing for the
climate emission prediction model.

Author: PLP Academy Student
Assignment: Week 2 - AI for Sustainable Development
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ClimateDataPreprocessor:
    """
    A comprehensive data preprocessing class for climate emission data.
    
    This class handles:
    - Data generation (simulating World Bank Open Data)
    - Feature engineering
    - Data cleaning and normalization
    - Train/test splitting
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor with random state for reproducibility.
        
        Args:
            random_state (int): Random seed for consistent results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
        np.random.seed(random_state)
        
    def generate_climate_dataset(self, n_countries=195, start_year=2000, end_year=2023):
        """
        Generate a comprehensive climate dataset based on real-world patterns.
        
        This simulates data from sources like:
        - World Bank Open Data
        - UN Statistics Division
        - International Energy Agency
        
        Args:
            n_countries (int): Number of countries to simulate
            start_year (int): Starting year for data
            end_year (int): Ending year for data
            
        Returns:
            pd.DataFrame: Generated climate dataset
        """
        print(f"🌍 Generating climate dataset for {n_countries} countries ({start_year}-{end_year})")
        
        countries = [f"Country_{i:03d}" for i in range(1, n_countries + 1)]
        years = list(range(start_year, end_year + 1))
        
        data = []
        
        for country in countries:
            # Assign country characteristics based on realistic distributions
            base_gdp = np.random.uniform(1000, 80000)  # GDP per capita range
            base_population = np.random.uniform(50, 1500)  # Population density
            development_level = 'Developed' if base_gdp > 25000 else 'Developing'
            
            # Regional characteristics
            region = np.random.choice(['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania'])
            
            for year in years:
                # Add temporal trends (climate action improving over time)
                year_factor = (year - start_year) / (end_year - start_year)
                
                # Economic indicators with realistic growth patterns
                gdp_per_capita = base_gdp * (1 + np.random.normal(0.02, 0.05)) * (1 + year_factor * 0.3)
                population_density = base_population * (1 + year_factor * 0.2 + np.random.normal(0, 0.02))
                
                # Energy and industrial indicators
                energy_consumption = gdp_per_capita * 0.3 + np.random.normal(0, 100)
                industrial_activity = gdp_per_capita * 0.25 + np.random.normal(0, 80)
                
                # Renewable energy adoption (improving over time)
                renewable_energy_pct = min(80, max(5, 15 + year_factor * 25 + np.random.normal(0, 5)))
                
                # Urban development
                urban_population_pct = min(95, max(20, 45 + year_factor * 15 + np.random.normal(0, 3)))
                transport_emissions = urban_population_pct * 2 + np.random.normal(0, 10)
                
                # Environmental policies (improving over time)
                policy_score = min(100, max(0, 30 + year_factor * 40 + np.random.normal(0, 8)))
                
                # Calculate CO2 emissions based on realistic relationships
                co2_emissions = (
                    gdp_per_capita * 0.0002 +      # Economic activity impact
                    energy_consumption * 0.01 +     # Energy consumption impact
                    industrial_activity * 0.008 +   # Industrial processes
                    transport_emissions * 0.05 +    # Transportation sector
                    population_density * 0.002 -    # Population density effect
                    renewable_energy_pct * 0.1 -    # Renewable energy benefit
                    policy_score * 0.02 +           # Policy effectiveness
                    np.random.normal(0, 1)          # Random variation
                )
                
                # Ensure realistic bounds for CO2 emissions
                co2_emissions = max(0.5, min(50, co2_emissions))
                
                data.append({
                    'Country': country,
                    'Year': year,
                    'Region': region,
                    'GDP_per_capita': round(gdp_per_capita, 2),
                    'Population_density': round(population_density, 2),
                    'Energy_consumption': round(energy_consumption, 2),
                    'Industrial_activity': round(industrial_activity, 2),
                    'Renewable_energy_pct': round(renewable_energy_pct, 2),
                    'Urban_population_pct': round(urban_population_pct, 2),
                    'Transport_emissions': round(transport_emissions, 2),
                    'Policy_score': round(policy_score, 2),
                    'Development_level': development_level,
                    'CO2_emissions': round(co2_emissions, 3)
                })
        
        df = pd.DataFrame(data)
        print(f"✅ Dataset generated successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"🌍 Countries: {df['Country'].nunique()}")
        print(f"📅 Years: {df['Year'].min()} - {df['Year'].max()}")
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features through feature engineering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("🔧 Engineering additional features...")
        
        df_engineered = df.copy()
        
        # Economic efficiency features
        df_engineered['GDP_Energy_ratio'] = df_engineered['GDP_per_capita'] / (df_engineered['Energy_consumption'] + 1)
        df_engineered['GDP_Industrial_ratio'] = df_engineered['GDP_per_capita'] / (df_engineered['Industrial_activity'] + 1)
        
        # Environmental features
        df_engineered['Renewable_ratio'] = df_engineered['Renewable_energy_pct'] / 100
        df_engineered['Carbon_intensity'] = df_engineered['CO2_emissions'] / (df_engineered['GDP_per_capita'] + 1)
        
        # Urban development features
        df_engineered['Urban_density'] = df_engineered['Urban_population_pct'] * df_engineered['Population_density'] / 100
        df_engineered['Transport_per_capita'] = df_engineered['Transport_emissions'] / (df_engineered['Population_density'] + 1)
        
        # Policy effectiveness features
        df_engineered['Policy_effectiveness'] = df_engineered['Policy_score'] * df_engineered['Renewable_ratio']
        df_engineered['Green_development'] = df_engineered['GDP_per_capita'] * df_engineered['Renewable_ratio']
        
        # Temporal features
        df_engineered['Years_since_2000'] = df_engineered['Year'] - 2000
        df_engineered['Decade'] = (df_engineered['Year'] // 10) * 10
        
        print(f"✅ Feature engineering completed!")
        print(f"📈 New features added: {len(df_engineered.columns) - len(df.columns)}")
        
        return df_engineered
    
    def preprocess_data(self, df, target_column='CO2_emissions'):
        """
        Complete preprocessing pipeline for the climate dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target variable
            
        Returns:
            tuple: (X_processed, y, feature_columns)
        """
        print("🔄 Starting data preprocessing pipeline...")
        
        # Engineer features
        df_processed = self.engineer_features(df)
        
        # Encode categorical variables
        df_processed['Development_level_encoded'] = self.label_encoder.fit_transform(df_processed['Development_level'])
        
        # Create region dummy variables
        region_dummies = pd.get_dummies(df_processed['Region'], prefix='Region')
        df_processed = pd.concat([df_processed, region_dummies], axis=1)
        
        # Select features for modeling
        self.feature_columns = [
            'GDP_per_capita', 'Population_density', 'Energy_consumption', 'Industrial_activity',
            'Renewable_energy_pct', 'Urban_population_pct', 'Transport_emissions', 'Policy_score',
            'Development_level_encoded', 'GDP_Energy_ratio', 'GDP_Industrial_ratio',
            'Urban_density', 'Policy_effectiveness', 'Green_development', 'Years_since_2000'
        ]
        
        # Add region dummy columns
        region_columns = [col for col in df_processed.columns if col.startswith('Region_')]
        self.feature_columns.extend(region_columns)
        
        # Prepare features and target
        X = df_processed[self.feature_columns]
        y = df_processed[target_column]
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        X_processed = pd.DataFrame(X_imputed, columns=self.feature_columns)
        
        print(f"✅ Preprocessing completed!")
        print(f"📊 Features: {len(self.feature_columns)}")
        print(f"🎯 Target: {target_column}")
        print(f"📈 Samples: {len(X_processed)}")
        
        return X_processed, y, self.feature_columns
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) or X_train_scaled if X_test is None
        """
        print("📏 Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"✅ Features scaled for training and test sets")
            return X_train_scaled, X_test_scaled
        else:
            print(f"✅ Features scaled for training set")
            return X_train_scaled
    
    def get_data_summary(self, df):
        """
        Generate a comprehensive summary of the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Summary statistics and information
        """
        summary = {
            'shape': df.shape,
            'countries': df['Country'].nunique(),
            'years': f"{df['Year'].min()}-{df['Year'].max()}",
            'regions': df['Region'].nunique() if 'Region' in df.columns else 0,
            'developed_countries': len(df[df['Development_level'] == 'Developed']['Country'].unique()),
            'developing_countries': len(df[df['Development_level'] == 'Developing']['Country'].unique()),
            'avg_emissions': df['CO2_emissions'].mean(),
            'max_emissions': df['CO2_emissions'].max(),
            'min_emissions': df['CO2_emissions'].min(),
            'missing_values': df.isnull().sum().sum()
        }
        
        return summary

def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    print("🌍 ClimateAI Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = ClimateDataPreprocessor(random_state=42)
    
    # Generate dataset
    df = preprocessor.generate_climate_dataset(n_countries=195, start_year=2000, end_year=2023)
    
    # Get data summary
    summary = preprocessor.get_data_summary(df)
    print(f"\n📊 Dataset Summary:")
    for key, value in summary.items():
        print(f"   • {key}: {value}")
    
    # Preprocess data
    X, y, feature_columns = preprocessor.preprocess_data(df)
    
    print(f"\n🎯 Ready for model training!")
    print(f"📈 Features: {len(feature_columns)}")
    print(f"📊 Samples: {len(X)}")
    
    return df, X, y, feature_columns, preprocessor

if __name__ == "__main__":
    main()
