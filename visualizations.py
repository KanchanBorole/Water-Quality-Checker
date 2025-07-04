import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_correlation_heatmap(data):
    """Create a correlation heatmap for the dataset"""
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Matrix of Water Quality Parameters',
        xaxis_title='Parameters',
        yaxis_title='Parameters',
        height=600
    )
    
    return fig

def create_feature_distributions(data):
    """Create distribution plots for all features"""
    # Get numeric columns (excluding Potability)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Potability' in numeric_cols:
        numeric_cols.remove('Potability')
    
    # Create subplots
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numeric_cols,
        specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
    )
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=data[col],
                name=col,
                nbinsx=30,
                showlegend=False
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title='Distribution of Water Quality Parameters',
        height=400 * n_rows,
        showlegend=False
    )
    
    return fig

def create_prediction_gauge(confidence, prediction):
    """Create a gauge chart for prediction confidence"""
    # Set color based on prediction
    if prediction == 1:
        color = "green"
        title_text = "POTABLE"
    else:
        color = "red"
        title_text = "NOT POTABLE"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {title_text}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig

def create_parameter_comparison(data, parameter):
    """Create a comparison plot for a parameter by potability"""
    fig = px.box(
        data, 
        x='Potability', 
        y=parameter,
        color='Potability',
        title=f'{parameter} Distribution by Potability'
    )
    
    fig.update_xaxis(title='Potability (0: Not Potable, 1: Potable)')
    fig.update_yaxis(title=parameter)
    
    return fig

def create_scatter_matrix(data, dimensions):
    """Create scatter matrix for selected dimensions"""
    fig = px.scatter_matrix(
        data,
        dimensions=dimensions,
        color='Potability',
        title="Scatter Matrix of Water Quality Parameters"
    )
    
    fig.update_layout(height=800)
    
    return fig

def create_potability_pie_chart(data):
    """Create pie chart for potability distribution"""
    potability_counts = data['Potability'].value_counts()
    
    fig = px.pie(
        values=potability_counts.values,
        names=['Not Potable', 'Potable'],
        title='Water Potability Distribution',
        color_discrete_map={'Not Potable': 'red', 'Potable': 'green'}
    )
    
    return fig

def create_parameter_radar_chart(input_data, reference_data=None):
    """Create radar chart for water quality parameters"""
    parameters = list(input_data.keys())
    values = list(input_data.values())
    
    # Normalize values for radar chart (0-1 scale)
    if reference_data is not None:
        normalized_values = []
        for param in parameters:
            if param in reference_data.columns:
                min_val = reference_data[param].min()
                max_val = reference_data[param].max()
                normalized_val = (input_data[param] - min_val) / (max_val - min_val)
                normalized_values.append(max(0, min(1, normalized_val)))
            else:
                normalized_values.append(0.5)
    else:
        normalized_values = [0.5] * len(parameters)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=parameters,
        fill='toself',
        name='Current Sample'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Water Quality Parameters Radar Chart"
    )
    
    return fig
