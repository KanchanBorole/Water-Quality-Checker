import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_animated_water_drop():
    """Create animated water drop visualization"""
    return """
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <div class="water-drop">
            <div class="water-drop-inner"></div>
        </div>
    </div>
    <style>
        .water-drop {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #00D4AA 0%, #2E8B57 100%);
            border-radius: 50% 50% 50% 0;
            transform: rotate(-45deg);
            position: relative;
            animation: drop-bounce 2s infinite;
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.5);
        }
        
        .water-drop-inner {
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50% 50% 50% 0;
            position: absolute;
            top: 10px;
            left: 10px;
            transform: rotate(45deg);
        }
        
        @keyframes drop-bounce {
            0%, 100% { transform: rotate(-45deg) translateY(0); }
            50% { transform: rotate(-45deg) translateY(-10px); }
        }
    </style>
    """

def create_quality_score_animation(score):
    """Create animated quality score visualization"""
    if score >= 90:
        color = "#00D4AA"
        grade = "A+"
        emoji = "🏆"
    elif score >= 80:
        color = "#2E8B57"
        grade = "A"
        emoji = "✅"
    elif score >= 70:
        color = "#FFB347"
        grade = "B"
        emoji = "⚠️"
    elif score >= 60:
        color = "#FF8C00"
        grade = "C"
        emoji = "🔔"
    else:
        color = "#FF6B6B"
        grade = "F"
        emoji = "❌"
    
    return f"""
    <div class="quality-score-container">
        <div class="quality-score" style="background: {color};">
            <div class="score-emoji">{emoji}</div>
            <div class="score-value">{score}</div>
            <div class="score-grade">{grade}</div>
        </div>
    </div>
    <style>
        .quality-score-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        
        .quality-score {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-weight: bold;
            position: relative;
            animation: score-pulse 2s infinite;
            box-shadow: 0 0 30px rgba(0, 212, 170, 0.4);
        }}
        
        .score-emoji {{
            font-size: 2rem;
            margin-bottom: 5px;
        }}
        
        .score-value {{
            font-size: 1.8rem;
            line-height: 1;
        }}
        
        .score-grade {{
            font-size: 1.2rem;
            margin-top: 5px;
        }}
        
        @keyframes score-pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
    </style>
    """

def create_live_statistics_card(title, value, unit, trend=None):
    """Create animated statistics card"""
    trend_icon = ""
    trend_color = "#00D4AA"
    
    if trend:
        if trend > 0:
            trend_icon = "📈"
            trend_color = "#00D4AA"
        elif trend < 0:
            trend_icon = "📉"
            trend_color = "#FF6B6B"
        else:
            trend_icon = "➡️"
            trend_color = "#FFB347"
    
    return f"""
    <div class="live-stats-card">
        <div class="stats-header">
            <h3>{title}</h3>
            {f'<span class="trend-indicator" style="color: {trend_color};">{trend_icon}</span>' if trend is not None else ''}
        </div>
        <div class="stats-value">
            <span class="value-number">{value}</span>
            <span class="value-unit">{unit}</span>
        </div>
    </div>
    <style>
        .live-stats-card {{
            background: linear-gradient(145deg, #1A2332 0%, #2D3B4A 100%);
            border: 1px solid #00D4AA;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 10px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .live-stats-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 212, 170, 0.3);
        }}
        
        .live-stats-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 170, 0.1), transparent);
            animation: shimmer 3s infinite;
        }}
        
        .stats-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .stats-header h3 {{
            color: #00D4AA;
            margin: 0;
            font-size: 1.1rem;
        }}
        
        .trend-indicator {{
            font-size: 1.2rem;
        }}
        
        .stats-value {{
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 0.5rem;
        }}
        
        .value-number {{
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
            text-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
        }}
        
        .value-unit {{
            font-size: 1rem;
            color: #FFB347;
        }}
        
        @keyframes shimmer {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
    </style>
    """

def create_interactive_parameter_slider():
    """Create interactive parameter adjustment interface"""
    st.markdown("""
    <div class="parameter-slider-container">
        <h3>🎛️ Interactive Parameter Adjustment</h3>
        <p>Adjust parameters in real-time to see how they affect water quality predictions</p>
    </div>
    """, unsafe_allow_html=True)

def create_prediction_confidence_bar(confidence, prediction):
    """Create animated confidence bar"""
    if prediction == 1:
        bar_color = "#00D4AA"
        bg_color = "rgba(0, 212, 170, 0.1)"
    else:
        bar_color = "#FF6B6B"
        bg_color = "rgba(255, 107, 107, 0.1)"
    
    return f"""
    <div class="confidence-bar-container">
        <div class="confidence-label">Prediction Confidence</div>
        <div class="confidence-bar-bg" style="background: {bg_color};">
            <div class="confidence-bar-fill" style="width: {confidence}%; background: {bar_color};">
                <div class="confidence-percentage">{confidence:.1f}%</div>
            </div>
        </div>
    </div>
    <style>
        .confidence-bar-container {{
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(145deg, #1A2332 0%, #2D3B4A 100%);
            border-radius: 10px;
            border: 1px solid #00D4AA;
        }}
        
        .confidence-label {{
            color: #00D4AA;
            font-size: 1.1rem;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        .confidence-bar-bg {{
            height: 30px;
            border-radius: 15px;
            position: relative;
            overflow: hidden;
        }}
        
        .confidence-bar-fill {{
            height: 100%;
            border-radius: 15px;
            position: relative;
            animation: fill-animation 2s ease-out;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .confidence-percentage {{
            color: white;
            font-weight: bold;
            font-size: 0.9rem;
            text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
        }}
        
        @keyframes fill-animation {{
            0% {{ width: 0%; }}
            100% {{ width: {confidence}%; }}
        }}
    </style>
    """

def create_real_time_chart(data, parameter):
    """Create real-time updating chart"""
    # Simulate real-time data updates
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = data.sample(50).reset_index(drop=True)
    
    fig = px.line(
        st.session_state.chart_data,
        y=parameter,
        title=f"📊 Real-time {parameter} Monitoring",
        color_discrete_sequence=['#00D4AA']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=16, color='#00D4AA'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

def create_notification_system():
    """Create notification system for alerts"""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    # Display notifications
    for notification in st.session_state.notifications[-3:]:  # Show last 3 notifications
        alert_type = notification.get('type', 'info')
        message = notification.get('message', '')
        timestamp = notification.get('timestamp', '')
        
        if alert_type == 'success':
            icon = "✅"
            color = "#00D4AA"
        elif alert_type == 'warning':
            icon = "⚠️"
            color = "#FFB347"
        elif alert_type == 'error':
            icon = "❌"
            color = "#FF6B6B"
        else:
            icon = "ℹ️"
            color = "#2E8B57"
        
        st.markdown(f"""
        <div class="notification-item" style="border-left-color: {color};">
            <div class="notification-icon" style="color: {color};">{icon}</div>
            <div class="notification-content">
                <div class="notification-message">{message}</div>
                <div class="notification-timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def add_notification(message, alert_type='info'):
    """Add notification to the system"""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notification = {
        'message': message,
        'type': alert_type,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    
    st.session_state.notifications.append(notification)
    
    # Keep only last 10 notifications
    if len(st.session_state.notifications) > 10:
        st.session_state.notifications = st.session_state.notifications[-10:]

# Add notification styles
notification_styles = """
<style>
.notification-item {
    background: linear-gradient(145deg, #1A2332 0%, #2D3B4A 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-left: 4px solid #00D4AA;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slide-in 0.5s ease-out;
}

.notification-icon {
    font-size: 1.2rem;
    min-width: 24px;
}

.notification-content {
    flex: 1;
}

.notification-message {
    color: white;
    font-size: 0.9rem;
    margin-bottom: 4px;
}

.notification-timestamp {
    color: #FFB347;
    font-size: 0.8rem;
    opacity: 0.8;
}

@keyframes slide-in {
    0% { transform: translateX(-100%); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}
</style>
"""