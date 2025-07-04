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
from interactive_features import (
    create_animated_water_drop, 
    create_quality_score_animation, 
    create_live_statistics_card,
    create_prediction_confidence_bar,
    add_notification,
    create_notification_system,
    notification_styles
)

# Page configuration
st.set_page_config(
    page_title="AI Water Quality Analyzer",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #00D4AA 0%, #2E8B57 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1A2332 0%, #2D3B4A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #00D4AA;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 212, 170, 0.2);
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00D4AA;
        background: linear-gradient(90deg, rgba(0, 212, 170, 0.1) 0%, rgba(26, 35, 50, 0.5) 100%);
    }
    
    .prediction-result {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .potable-result {
        background: linear-gradient(135deg, #00D4AA 0%, #2E8B57 100%);
        color: white;
        border: 2px solid #00D4AA;
    }
    
    .not-potable-result {
        background: linear-gradient(135deg, #FF6B6B 0%, #CC5A5A 100%);
        color: white;
        border: 2px solid #FF6B6B;
    }
    
    .parameter-info {
        background: linear-gradient(145deg, #1A2332 0%, #2D3B4A 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #00D4AA;
    }
    
    .sidebar .stSelectbox > label {
        color: #00D4AA;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #2E8B57 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
    }
    
    .animated-icon {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(46, 139, 87, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 170, 0.3);
        margin: 1rem 0;
    }
    
    .data-quality-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .excellent { background: #00D4AA; color: white; }
    .good { background: #2E8B57; color: white; }
    .warning { background: #FFB347; color: black; }
    .danger { background: #FF6B6B; color: white; }
</style>
""", unsafe_allow_html=True)

# Add notification styles and interactive features
st.markdown(notification_styles, unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = None

# Enhanced Title and description with animation
st.markdown("""
<div class="main-header">
    <h1 class="animated-icon">🚰 AI-Powered Water Quality Analyzer</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem;">
        Advanced machine learning technology to predict water safety and quality
    </p>
    <div style="margin-top: 1rem;">
        <span class="data-quality-indicator excellent">🔬 Smart Analysis</span>
        <span class="data-quality-indicator good">🤖 AI Predictions</span>
        <span class="data-quality-indicator warning">📊 Interactive Charts</span>
        <span class="data-quality-indicator danger">⚡ Real-time Results</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar for navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #00D4AA 0%, #2E8B57 100%); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🌊 Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔍 Prediction", "📈 Dashboard", "📋 History"],
    help="Navigate through different sections of the water quality analyzer"
)

# Add interactive sidebar info
st.sidebar.markdown("""
<div class="feature-highlight">
    <h4>💡 Quick Tips</h4>
    <ul>
        <li>Start with <strong>Data Analysis</strong> to explore your data</li>
        <li>Train models in <strong>Model Training</strong></li>
        <li>Make predictions in <strong>Prediction</strong> section</li>
        <li>View results in <strong>Dashboard</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Initialize data handler
@st.cache_resource
def get_data_handler():
    return DataHandler()

data_handler = get_data_handler()

# Home Page
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(create_animated_water_drop(), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <h2>🌊 About Water Quality Analysis</h2>
            <h3>💡 Why Water Quality Matters</h3>
            <p>Access to clean drinking water is a fundamental human right. This AI tool helps predict whether water is safe to drink based on easily measurable chemical properties.</p>
            
            <h3>🔬 Parameters We Analyze</h3>
            <ul>
                <li><strong>pH</strong>: Acidity/alkalinity level (6.5-8.5 is ideal)</li>
                <li><strong>Hardness</strong>: Calcium and magnesium content</li>
                <li><strong>Solids</strong>: Total dissolved solids (TDS)</li>
                <li><strong>Chloramines</strong>: Disinfectant levels</li>
                <li><strong>Sulfate</strong>: Sulfate ion concentration</li>
                <li><strong>Conductivity</strong>: Electrical conductivity</li>
                <li><strong>Organic Carbon</strong>: Total organic carbon</li>
                <li><strong>Trihalomethanes</strong>: Disinfection byproducts</li>
                <li><strong>Turbidity</strong>: Water clarity measure</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <h3>📊 Quick Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if data_handler.data is not None:
            total_samples = len(data_handler.data)
            potable_samples = data_handler.data['Potability'].sum()
            potability_rate = potable_samples/total_samples*100
            
            # Enhanced metrics with custom styling
            col2a, col2b = st.columns(2)
            with col2a:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: #00D4AA; margin: 0;">{total_samples}</h2>
                    <p style="margin: 0;">Total Samples</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: #2E8B57; margin: 0;">{potable_samples}</h2>
                    <p style="margin: 0;">Potable Samples</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2b:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: #FFB347; margin: 0;">{potability_rate:.1f}%</h2>
                    <p style="margin: 0;">Potability Rate</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Water quality indicator
                if potability_rate >= 70:
                    quality_color = "#00D4AA"
                    quality_text = "Excellent"
                elif potability_rate >= 50:
                    quality_color = "#2E8B57"
                    quality_text = "Good"
                else:
                    quality_color = "#FF6B6B"
                    quality_text = "Needs Attention"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: {quality_color}; margin: 0;">{quality_text}</h2>
                    <p style="margin: 0;">Overall Quality</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <p>🔄 Load data to see detailed statistics and insights</p>
            </div>
            """, unsafe_allow_html=True)

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
                            
                            # Enhanced prediction result with custom styling
                            if prediction == 1:
                                result_class = "potable-result"
                                result_icon = "🟢"
                                result_message = "Water is SAFE to drink!"
                            else:
                                result_class = "not-potable-result"
                                result_icon = "🔴"
                                result_message = "Water is NOT SAFE to drink!"
                            
                            st.markdown(f"""
                            <div class="prediction-result {result_class}">
                                <div style="font-size: 3rem; margin-bottom: 1rem;">{result_icon}</div>
                                <div style="font-size: 2rem; margin-bottom: 1rem;">{result_text}</div>
                                <div style="font-size: 1.2rem; margin-bottom: 1rem;">{result_message}</div>
                                <div style="font-size: 1.5rem;">Confidence: {confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create enhanced gauge chart
                            fig_gauge = create_prediction_gauge(confidence, prediction)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Parameter Analysis")
                            
                            # Enhanced parameter analysis with custom styling
                            parameter_info = get_water_quality_info(input_data)
                            st.markdown(f"""
                            <div class="parameter-info">
                                {parameter_info.replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add interactive parameter radar chart
                            from visualizations import create_parameter_radar_chart
                            fig_radar = create_parameter_radar_chart(input_data, data_handler.data)
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Enhanced confidence bar
                        st.markdown(create_prediction_confidence_bar(confidence, prediction), unsafe_allow_html=True)
                        
                        # Add to history
                        prediction_record = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'prediction': 'Potable' if prediction == 1 else 'Not Potable',
                            'confidence': confidence,
                            **input_data
                        }
                        st.session_state.prediction_history.append(prediction_record)
                        
                        # Add notification
                        if prediction == 1:
                            add_notification(f"✅ Water sample predicted as POTABLE with {confidence:.1f}% confidence", "success")
                        else:
                            add_notification(f"❌ Water sample predicted as NOT POTABLE with {confidence:.1f}% confidence", "warning")
                        
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
    st.header("📋 Prediction History")
    
    # Add notification system
    with st.expander("🔔 Recent Notifications", expanded=False):
        create_notification_system()
    
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
