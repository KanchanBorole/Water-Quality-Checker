import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px
from data_handler import DataHandler

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_test_proba = None
        self.data_handler = DataHandler()
        self.data_handler.data = data
    
    def train_model(self, model_type, test_size=0.2, **kwargs):
        """Train a machine learning model"""
        
        # Preprocess data
        X, y = self.data_handler.preprocess_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Select and configure model
        if model_type == "Random Forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == "Gradient Boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif model_type == "Support Vector Machine":
            self.model = SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=42
            )
        elif model_type == "Logistic Regression":
            self.model = LogisticRegression(
                C=kwargs.get('C', 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        self.y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_test_proba)
        }
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        return self.model, metrics, feature_importance
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        feature_names = self.X_train.columns
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        feature_importance = pd.Series(importances, index=feature_names)
        return feature_importance.sort_values(ascending=True)
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        if self.y_test is None or self.y_test_proba is None:
            return None
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_test_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_test_proba)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        return fig
    
    def predict(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Transform new data
        new_data_processed = self.data_handler.transform_new_data(new_data)
        
        # Make prediction
        prediction = self.model.predict(new_data_processed)
        prediction_proba = self.model.predict_proba(new_data_processed)
        
        return prediction, prediction_proba
