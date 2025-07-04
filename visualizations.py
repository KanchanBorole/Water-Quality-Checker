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
    
    # Create enhanced heatmap with custom color scheme
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, '#FF6B6B'], [0.5, '#1A2332'], [1, '#00D4AA']],
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '🔥 Correlation Matrix of Water Quality Parameters',
            'x': 0.5,
            'font': {'size': 20, 'color': '#00D4AA'}
        },
        xaxis_title='Parameters',
        yaxis_title='Parameters',
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
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
    
    # Enhanced color palette
    colors = ['#00D4AA', '#2E8B57', '#FFB347', '#FF6B6B', '#8A2BE2', '#FF1493', '#32CD32', '#FF4500', '#1E90FF']
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        # Create enhanced histogram with custom styling
        fig.add_trace(
            go.Histogram(
                x=data[col],
                name=col,
                nbinsx=30,
                showlegend=False,
                marker=dict(
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=f'<b>{col}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title={
            'text': '📊 Distribution of Water Quality Parameters',
            'x': 0.5,
            'font': {'size': 20, 'color': '#00D4AA'}
        },
        height=400 * n_rows,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title_font=dict(color='white'))
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', title_font=dict(color='white'))
    
    return fig

def create_prediction_gauge(confidence, prediction):
    """Create a gauge chart for prediction confidence"""
    # Set enhanced colors based on prediction
    if prediction == 1:
        color = "#00D4AA"
        title_text = "POTABLE"
        steps_colors = [
            {'range': [0, 30], 'color': "#FF6B6B"},
            {'range': [30, 60], 'color': "#FFB347"},
            {'range': [60, 80], 'color': "#2E8B57"},
            {'range': [80, 100], 'color': "#00D4AA"}
        ]
    else:
        color = "#FF6B6B"
        title_text = "NOT POTABLE"
        steps_colors = [
            {'range': [0, 30], 'color': "#FF6B6B"},
            {'range': [30, 60], 'color': "#FFB347"},
            {'range': [60, 80], 'color': "#2E8B57"},
            {'range': [80, 100], 'color': "#00D4AA"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"🎯 Confidence: {title_text}",
            'font': {'size': 18, 'color': 'white'}
        },
        delta = {'reference': 50, 'increasing': {'color': color}},
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "white"
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': steps_colors,
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number = {'font': {'color': 'white', 'size': 24}}
    ))
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
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
    
    # Add current sample trace with enhanced styling
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=parameters,
        fill='toself',
        name='Current Sample',
        line=dict(color='#00D4AA', width=3),
        fillcolor='rgba(0, 212, 170, 0.3)',
        hovertemplate='<b>%{theta}</b><br>Normalized Value: %{r:.3f}<extra></extra>'
    ))
    
    # Add reference ranges if available
    if reference_data is not None:
        # Add average reference trace
        avg_normalized = []
        for param in parameters:
            if param in reference_data.columns:
                avg_val = reference_data[param].mean()
                min_val = reference_data[param].min()
                max_val = reference_data[param].max()
                avg_normalized_val = (avg_val - min_val) / (max_val - min_val)
                avg_normalized.append(max(0, min(1, avg_normalized_val)))
            else:
                avg_normalized.append(0.5)
        
        fig.add_trace(go.Scatterpolar(
            r=avg_normalized,
            theta=parameters,
            fill='toself',
            name='Dataset Average',
            line=dict(color='#FFB347', width=2, dash='dash'),
            fillcolor='rgba(255, 179, 71, 0.1)',
            hovertemplate='<b>%{theta}</b><br>Average Value: %{r:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={
            'text': '🎯 Water Quality Parameters Radar Chart',
            'x': 0.5,
            'font': {'size': 18, 'color': '#00D4AA'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    return fig
