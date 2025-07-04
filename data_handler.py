import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st

class DataHandler:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = None
        self.imputer = None
        self.load_default_data()
    
    def load_default_data(self):
        """Load default water quality dataset"""
        try:
            self.data = pd.read_csv('data/water_potability.csv')
            st.success("Default water quality dataset loaded successfully!")
        except FileNotFoundError:
            # Create synthetic data if file doesn't exist
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic water quality data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic data based on realistic water quality parameters
        n_samples = 2000
        
        # Create correlated features
        data = {}
        
        # pH (6.5-8.5 for potable, wider range for non-potable)
        potability = np.random.binomial(1, 0.6, n_samples)
        
        data['ph'] = np.where(
            potability == 1,
            np.random.normal(7.2, 0.5, n_samples),
            np.random.normal(6.8, 1.2, n_samples)
        )
        
        # Hardness (mg/L)
        data['Hardness'] = np.random.normal(200, 50, n_samples)
        
        # Solids (ppm)
        data['Solids'] = np.random.normal(20000, 5000, n_samples)
        
        # Chloramines (ppm)
        data['Chloramines'] = np.where(
            potability == 1,
            np.random.normal(7, 1, n_samples),
            np.random.normal(6, 2, n_samples)
        )
        
        # Sulfate (mg/L)
        data['Sulfate'] = np.random.normal(300, 100, n_samples)
        
        # Conductivity (μS/cm)
        data['Conductivity'] = np.random.normal(400, 100, n_samples)
        
        # Organic carbon (ppm)
        data['Organic_carbon'] = np.random.normal(14, 3, n_samples)
        
        # Trihalomethanes (μg/L)
        data['Trihalomethanes'] = np.where(
            potability == 1,
            np.random.normal(60, 15, n_samples),
            np.random.normal(80, 20, n_samples)
        )
        
        # Turbidity (NTU)
        data['Turbidity'] = np.where(
            potability == 1,
            np.random.normal(3, 1, n_samples),
            np.random.normal(5, 2, n_samples)
        )
        
        # Add some missing values
        for col in data.keys():
            missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
            data[col] = np.array(data[col])
            data[col][missing_indices] = np.nan
        
        data['Potability'] = potability
        
        self.data = pd.DataFrame(data)
        
        # Ensure realistic ranges
        self.data['ph'] = np.clip(self.data['ph'], 0, 14)
        self.data['Hardness'] = np.clip(self.data['Hardness'], 0, None)
        self.data['Solids'] = np.clip(self.data['Solids'], 0, None)
        self.data['Chloramines'] = np.clip(self.data['Chloramines'], 0, None)
        self.data['Sulfate'] = np.clip(self.data['Sulfate'], 0, None)
        self.data['Conductivity'] = np.clip(self.data['Conductivity'], 0, None)
        self.data['Organic_carbon'] = np.clip(self.data['Organic_carbon'], 0, None)
        self.data['Trihalomethanes'] = np.clip(self.data['Trihalomethanes'], 0, None)
        self.data['Turbidity'] = np.clip(self.data['Turbidity'], 0, None)
        
        st.info("Synthetic water quality dataset created for demonstration.")
    
    def load_custom_data(self, data):
        """Load custom dataset"""
        self.data = data
        self.processed_data = None
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Separate features and target
        X = self.data.drop('Potability', axis=1)
        y = self.data['Potability']
        
        # Handle missing values
        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Create processed dataframe
        self.processed_data = pd.DataFrame(X_scaled, columns=X.columns)
        
        return self.processed_data, y
    
    def transform_new_data(self, new_data):
        """Transform new data using fitted preprocessors"""
        if self.imputer is None or self.scaler is None:
            raise ValueError("Preprocessors not fitted. Call preprocess_data first.")
        
        # Convert to DataFrame if needed
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        # Apply same preprocessing
        new_data_imputed = self.imputer.transform(new_data)
        new_data_scaled = self.scaler.transform(new_data_imputed)
        
        return pd.DataFrame(new_data_scaled, columns=new_data.columns)
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        if self.data is None:
            return None
        
        summary = {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'potability_distribution': self.data['Potability'].value_counts().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        
        return summary
