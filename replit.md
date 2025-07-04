# AI Water Quality Analyzer

## Overview

This is a Streamlit-based web application that uses machine learning to predict water potability based on chemical parameters. The application provides a comprehensive suite of tools for data analysis, model training, and water quality prediction with interactive visualizations.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with expandable sidebar navigation
- **State Management**: Streamlit session state for maintaining user data and model state

### Backend Architecture
- **Data Processing**: Pandas and NumPy for data manipulation
- **Machine Learning**: Scikit-learn for model training and evaluation
- **Model Types**: Support for Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Data Preprocessing**: StandardScaler and SimpleImputer for feature scaling and missing value handling

### Application Structure
```
├── app.py                 # Main Streamlit application
├── data_handler.py        # Data loading and preprocessing
├── model_trainer.py       # ML model training and evaluation
├── visualizations.py      # Plotly chart creation
├── utils.py              # Utility functions and validation
└── data/
    └── water_potability.csv  # Default dataset
```

## Key Components

### 1. Data Handler (`data_handler.py`)
- Loads default water quality dataset or creates synthetic data
- Handles data preprocessing including scaling and imputation
- Manages feature engineering and data validation

### 2. Model Trainer (`model_trainer.py`)
- Supports multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- Handles train/test split and model evaluation
- Provides performance metrics and ROC curve analysis

### 3. Visualization Engine (`visualizations.py`)
- Creates correlation heatmaps for feature analysis
- Generates distribution plots for data exploration
- Provides prediction gauge visualization

### 4. Utilities (`utils.py`)
- Input validation for water quality parameters
- Safety range checking for drinking water standards
- Informative feedback about water quality status

### 5. Main Application (`app.py`)
- Multi-page Streamlit interface
- Session state management for model persistence
- Navigation between different analysis views

## Data Flow

1. **Data Loading**: Application loads default dataset or accepts user uploads
2. **Data Preprocessing**: Missing values handled, features scaled using StandardScaler
3. **Model Training**: User selects algorithm and hyperparameters, model trains on processed data
4. **Prediction**: New water samples processed through trained model
5. **Visualization**: Results displayed through interactive Plotly charts
6. **History Tracking**: Predictions stored in session state for analysis

## External Dependencies

### Core Libraries
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms and preprocessing
- `plotly`: Interactive visualization

### Data Requirements
- Primary dataset: `water_potability.csv` with 9 water quality parameters
- Fallback: Synthetic data generation if dataset unavailable
- Parameters: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity

## Deployment Strategy

### Local Development
- Run using `streamlit run app.py`
- Requires Python 3.7+ with pip-installed dependencies
- Data files should be placed in `data/` directory

### Production Considerations
- Application designed for containerization
- Session state used for model persistence (not suitable for multi-user production without modifications)
- File-based model storage using pickle for trained models

## Changelog

- July 04, 2025. Initial setup with basic water quality analysis features
- July 04, 2025. Enhanced UI with dark theme, interactive animations, and advanced visualizations
- July 04, 2025. Added PostgreSQL database integration for persistent data storage

## User Preferences

Preferred communication style: Simple, everyday language.
UI Preference: Colorful, interactive theme with animations and modern dark design.