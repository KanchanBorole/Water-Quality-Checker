import pandas as pd
import numpy as np

def validate_input(input_data):
    """Validate input water quality parameters"""
    warnings = []
    
    # Define acceptable ranges for water quality parameters
    ranges = {
        'ph': (0, 14),
        'Hardness': (0, 1000),
        'Solids': (0, 100000),
        'Chloramines': (0, 20),
        'Sulfate': (0, 1000),
        'Conductivity': (0, 2000),
        'Organic_carbon': (0, 50),
        'Trihalomethanes': (0, 200),
        'Turbidity': (0, 20)
    }
    
    # Check if values are within acceptable ranges
    for param, value in input_data.items():
        if param in ranges:
            min_val, max_val = ranges[param]
            if value < min_val or value > max_val:
                warnings.append(f"{param} value {value} is outside typical range ({min_val}-{max_val})")
    
    # Check for potentially dangerous combinations
    if input_data['ph'] < 6.5 or input_data['ph'] > 8.5:
        warnings.append("pH outside safe drinking water range (6.5-8.5)")
    
    if input_data['Turbidity'] > 4:
        warnings.append("High turbidity may indicate contamination")
    
    if input_data['Trihalomethanes'] > 100:
        warnings.append("High trihalomethanes level may pose health risks")
    
    return {
        'valid': len(warnings) == 0,
        'warnings': warnings
    }

def get_water_quality_info(input_data):
    """Get informative text about water quality parameters"""
    info = []
    
    # pH analysis
    ph = input_data['ph']
    if ph < 6.5:
        info.append("🔴 **pH**: Too acidic - may cause corrosion")
    elif ph > 8.5:
        info.append("🔴 **pH**: Too alkaline - may cause scaling")
    else:
        info.append("🟢 **pH**: Within safe range")
    
    # Turbidity analysis
    turbidity = input_data['Turbidity']
    if turbidity > 4:
        info.append("🔴 **Turbidity**: High - water appears cloudy")
    elif turbidity > 1:
        info.append("🟡 **Turbidity**: Moderate - slightly cloudy")
    else:
        info.append("🟢 **Turbidity**: Low - clear water")
    
    # Hardness analysis
    hardness = input_data['Hardness']
    if hardness < 60:
        info.append("🟡 **Hardness**: Soft water")
    elif hardness < 120:
        info.append("🟢 **Hardness**: Moderately hard")
    elif hardness < 180:
        info.append("🟡 **Hardness**: Hard water")
    else:
        info.append("🔴 **Hardness**: Very hard water")
    
    # Chloramines analysis
    chloramines = input_data['Chloramines']
    if chloramines < 0.5:
        info.append("🔴 **Chloramines**: Too low - inadequate disinfection")
    elif chloramines > 4:
        info.append("🔴 **Chloramines**: Too high - may cause taste/odor issues")
    else:
        info.append("🟢 **Chloramines**: Adequate disinfection level")
    
    return "\n".join(info)

def get_parameter_description(parameter):
    """Get description of water quality parameters"""
    descriptions = {
        'ph': "pH measures the acidity or alkalinity of water. Safe drinking water should have a pH between 6.5 and 8.5.",
        'Hardness': "Water hardness is caused by dissolved calcium and magnesium. Measured in mg/L as CaCO3.",
        'Solids': "Total dissolved solids (TDS) represent the total amount of mobile charged ions dissolved in water.",
        'Chloramines': "Chloramines are disinfectants used to treat drinking water. They help prevent bacterial growth.",
        'Sulfate': "Sulfate is a naturally occurring ion found in most water supplies. High levels can cause laxative effects.",
        'Conductivity': "Electrical conductivity measures water's ability to conduct electricity, indicating ion concentration.",
        'Organic_carbon': "Total organic carbon (TOC) measures the amount of carbon in organic compounds in water.",
        'Trihalomethanes': "Trihalomethanes are chemical compounds formed when chlorine reacts with organic matter.",
        'Turbidity': "Turbidity measures water clarity. High turbidity can indicate the presence of disease-causing organisms."
    }
    return descriptions.get(parameter, "No description available.")

def calculate_water_quality_index(input_data):
    """Calculate a simple water quality index"""
    # Simple scoring system (0-100)
    score = 100
    
    # pH penalty
    ph = input_data['ph']
    if ph < 6.5 or ph > 8.5:
        score -= 20
    elif ph < 7.0 or ph > 8.0:
        score -= 10
    
    # Turbidity penalty
    turbidity = input_data['Turbidity']
    if turbidity > 4:
        score -= 25
    elif turbidity > 1:
        score -= 10
    
    # Trihalomethanes penalty
    thm = input_data['Trihalomethanes']
    if thm > 100:
        score -= 20
    elif thm > 80:
        score -= 10
    
    # Hardness consideration
    hardness = input_data['Hardness']
    if hardness > 300:
        score -= 10
    
    # Ensure score is between 0 and 100
    score = max(0, min(100, score))
    
    return score

def format_parameter_value(parameter, value):
    """Format parameter values with appropriate units"""
    units = {
        'ph': '',
        'Hardness': ' mg/L',
        'Solids': ' ppm',
        'Chloramines': ' ppm',
        'Sulfate': ' mg/L',
        'Conductivity': ' μS/cm',
        'Organic_carbon': ' ppm',
        'Trihalomethanes': ' μg/L',
        'Turbidity': ' NTU'
    }
    
    unit = units.get(parameter, '')
    return f"{value:.2f}{unit}"
