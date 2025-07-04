import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import pickle

from data_handler import DataHandler
from model_trainer import ModelTrainer
from visualizations import create_correlation_heatmap, create_feature_distributions, create_prediction_gauge
from utils import validate_input, get_water_quality_info

# Page configuration
st.set_page_config(
    page_title="AI Water Quality Analyzer",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = None

# Title and description
st.title("🚰 AI-Powered Water Quality Analyzer")
st.markdown("""
This application uses machine learning to predict water potability based on chemical parameters.
Upload your water quality data or use the built-in dataset to train models and make predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔍 Prediction", "📈 Dashboard", "📋 History"]
)

# Initialize data handler
@st.cache_resource
def get_data_handler():
    return DataHandler()

data_handler = get_data_handler()

# Home Page
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("About Water Quality Analysis")
        st.markdown("""
        ### Why Water Quality Matters
        Access to clean drinking water is a fundamental human right. This AI tool helps predict whether water is safe to drink based on easily measurable chemical properties.
        
        ### Parameters We Analyze
        - **pH**: Acidity/alkalinity level (6.5-8.5 is ideal)
        - **Hardness**: Calcium and magnesium content
        - **Solids**: Total dissolved solids (TDS)
        - **Chloramines**: Disinfectant levels
        - **Sulfate**: Sulfate ion concentration
        - **Conductivity**: Electrical conductivity
        - **Organic Carbon**: Total organic carbon
        - **Trihalomethanes**: Disinfection byproducts
        - **Turbidity**: Water clarity measure
        """)
    
    with col2:
        st.header("Quick Stats")
        if data_handler.data is not None:
            total_samples = len(data_handler.data)
            potable_samples = data_handler.data['Potability'].sum()
            st.metric("Total Samples", total_samples)
            st.metric("Potable Samples", potable_samples)
            st.metric("Potability Rate", f"{potable_samples/total_samples*100:.1f}%")
        else:
            st.info("Load data to see statistics")

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.header("Data Analysis")
    
    # Data upload option
    uploaded_file = st.file_uploader("Upload Water Quality Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            custom_data = pd.read_csv(uploaded_file)
            data_handler.load_custom_data(custom_data)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    if data_handler.data is not None:
        # Display data overview
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", data_handler.data.shape)
            st.write("**Missing Values:**")
            st.write(data_handler.data.isnull().sum())
        
        with col2:
            st.write("**Data Types:**")
            st.write(data_handler.data.dtypes)
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(data_handler.data.head())
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(data_handler.data.describe())
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Correlation heatmap
        st.plotly_chart(create_correlation_heatmap(data_handler.data), use_container_width=True)
        
        # Feature distributions
        st.plotly_chart(create_feature_distributions(data_handler.data), use_container_width=True)
        
        # Potability distribution
        potability_counts = data_handler.data['Potability'].value_counts()
        fig_pie = px.pie(
            values=potability_counts.values,
            names=['Not Potable', 'Potable'],
            title="Water Potability Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("Please upload data or the default dataset will be loaded automatically.")

# Model Training Page
elif page == "🤖 Model Training":
    st.header("Model Training")
    
    if data_handler.data is not None:
        # Model selection
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Gradient Boosting", "Support Vector Machine", "Logistic Regression"]
            )
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        
        # Training parameters
        st.subheader("Training Parameters")
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        elif model_type == "Support Vector Machine":
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            params = {'C': C, 'kernel': kernel}
        else:  # Logistic Regression
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            params = {'C': C}
        
        # Train model button
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    model_trainer = ModelTrainer(data_handler.data)
                    model, metrics, feature_importance = model_trainer.train_model(
                        model_type, test_size, **params
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.session_state.model_trainer = model_trainer
                    
                    st.success("Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1']:.3f}")
                    
                    # Feature importance
                    if feature_importance is not None:
                        st.subheader("Feature Importance")
                        fig_importance = px.bar(
                            x=feature_importance.values,
                            y=feature_importance.index,
                            orientation='h',
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # ROC Curve
                    if hasattr(model_trainer, 'y_test_proba'):
                        st.subheader("ROC Curve")
                        fig_roc = model_trainer.plot_roc_curve()
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    else:
        st.warning("Please load data first in the Data Analysis page.")

# Prediction Page
elif page == "🔍 Prediction":
    st.header("Water Quality Prediction")
    
    if st.session_state.model_trained and st.session_state.model is not None:
        st.subheader("Enter Water Quality Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=1.0)
                solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0, step=100.0)
            
            with col2:
                chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
                sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0, step=1.0)
                conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, step=1.0)
            
            with col3:
                organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=14.0, step=0.1)
                trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=66.0, step=0.1)
                turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)
            
            submitted = st.form_submit_button("Predict Water Quality", type="primary")
            
            if submitted:
                # Validate inputs
                input_data = {
                    'ph': ph, 'Hardness': hardness, 'Solids': solids,
                    'Chloramines': chloramines, 'Sulfate': sulfate,
                    'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
                    'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity
                }
                
                validation_result = validate_input(input_data)
                if validation_result["valid"]:
                    try:
                        # Make prediction
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.model.predict(input_df)[0]
                        prediction_proba = st.session_state.model.predict_proba(input_df)[0]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            result_text = "✅ POTABLE" if prediction == 1 else "❌ NOT POTABLE"
                            confidence = max(prediction_proba) * 100
                            
                            st.markdown(f"### {result_text}")
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            
                            # Create gauge chart
                            fig_gauge = create_prediction_gauge(confidence, prediction)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Parameter Analysis")
                            st.markdown(get_water_quality_info(input_data))
                        
                        # Add to history
                        prediction_record = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'prediction': 'Potable' if prediction == 1 else 'Not Potable',
                            'confidence': confidence,
                            **input_data
                        }
                        st.session_state.prediction_history.append(prediction_record)
                        
                        st.success("Prediction completed and saved to history!")
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                else:
                    st.error("Input validation failed:")
                    for warning in validation_result["warnings"]:
                        st.warning(warning)
    
    else:
        st.warning("Please train a model first in the Model Training page.")

# Dashboard Page
elif page == "📈 Dashboard":
    st.header("Water Quality Dashboard")
    
    if data_handler.data is not None:
        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_ph = data_handler.data['ph'].mean()
            st.metric("Average pH", f"{avg_ph:.2f}")
        
        with col2:
            avg_hardness = data_handler.data['Hardness'].mean()
            st.metric("Average Hardness", f"{avg_hardness:.1f} mg/L")
        
        with col3:
            avg_turbidity = data_handler.data['Turbidity'].mean()
            st.metric("Average Turbidity", f"{avg_turbidity:.2f} NTU")
        
        with col4:
            potability_rate = data_handler.data['Potability'].mean() * 100
            st.metric("Potability Rate", f"{potability_rate:.1f}%")
        
        # Parameter distributions by potability
        st.subheader("Parameter Analysis by Potability")
        
        parameters = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        selected_param = st.selectbox("Select Parameter", parameters)
        
        fig_box = px.box(
            data_handler.data,
            x='Potability',
            y=selected_param,
            title=f"{selected_param} Distribution by Potability"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Scatter plot matrix
        st.subheader("Parameter Relationships")
        selected_params = st.multiselect(
            "Select Parameters for Scatter Matrix",
            parameters,
            default=['ph', 'Hardness', 'Turbidity']
        )
        
        if len(selected_params) >= 2:
            fig_scatter = px.scatter_matrix(
                data_handler.data,
                dimensions=selected_params,
                color='Potability',
                title="Parameter Relationships"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    else:
        st.info("Please load data first in the Data Analysis page.")

# History Page
elif page == "📋 History":
    st.header("Prediction History")
    
    if st.session_state.prediction_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display summary
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            potable_count = len(history_df[history_df['prediction'] == 'Potable'])
            st.metric("Potable Predictions", potable_count)
        
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        
        # Display history table
        st.subheader("Prediction History")
        st.dataframe(history_df)
        
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name=f"water_quality_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()
    
    else:
        st.info("No predictions made yet. Use the Prediction page to make predictions.")

# Footer
st.markdown("---")
st.markdown("""
**AI Water Quality Analyzer** | Built with Streamlit | 
Promoting UN SDG Goal 6: Clean Water & Sanitation
""")
