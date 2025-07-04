import os
import psycopg2
import pandas as pd
from datetime import datetime
import json
import streamlit as st
from contextlib import contextmanager

class DatabaseHandler:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.host = os.getenv('PGHOST')
        self.port = os.getenv('PGPORT')
        self.database = os.getenv('PGDATABASE')
        self.user = os.getenv('PGUSER')
        self.password = os.getenv('PGPASSWORD')
        
        self.create_tables()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def create_tables(self):
        """Create necessary tables for the water quality analyzer"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ph FLOAT,
                        hardness FLOAT,
                        solids FLOAT,
                        chloramines FLOAT,
                        sulfate FLOAT,
                        conductivity FLOAT,
                        organic_carbon FLOAT,
                        trihalomethanes FLOAT,
                        turbidity FLOAT,
                        prediction INTEGER,
                        prediction_label VARCHAR(50),
                        confidence FLOAT,
                        model_type VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create datasets table for storing uploaded data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255),
                        description TEXT,
                        data_json TEXT,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        num_records INTEGER
                    );
                """)
                
                # Create model_performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id SERIAL PRIMARY KEY,
                        model_type VARCHAR(100),
                        accuracy FLOAT,
                        precision_score FLOAT,
                        recall FLOAT,
                        f1_score FLOAT,
                        roc_auc FLOAT,
                        training_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hyperparameters TEXT
                    );
                """)
                
                # Create user_sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255),
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_predictions INTEGER DEFAULT 0,
                        models_trained INTEGER DEFAULT 0
                    );
                """)
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Error creating database tables: {str(e)}")
    
    def save_prediction(self, input_data, prediction, confidence, model_type="Unknown"):
        """Save a prediction to the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                prediction_label = "Potable" if prediction == 1 else "Not Potable"
                
                cursor.execute("""
                    INSERT INTO predictions 
                    (ph, hardness, solids, chloramines, sulfate, conductivity, 
                     organic_carbon, trihalomethanes, turbidity, prediction, 
                     prediction_label, confidence, model_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    input_data.get('ph'),
                    input_data.get('Hardness'),
                    input_data.get('Solids'),
                    input_data.get('Chloramines'),
                    input_data.get('Sulfate'),
                    input_data.get('Conductivity'),
                    input_data.get('Organic_carbon'),
                    input_data.get('Trihalomethanes'),
                    input_data.get('Turbidity'),
                    prediction,
                    prediction_label,
                    confidence,
                    model_type
                ))
                
                prediction_id = cursor.fetchone()[0]
                conn.commit()
                return prediction_id
                
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")
            return None
    
    def get_prediction_history(self, limit=100):
        """Get prediction history from database"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT %s;
                """
                df = pd.read_sql_query(query, conn, params=[limit])
                return df
                
        except Exception as e:
            st.error(f"Error fetching prediction history: {str(e)}")
            return pd.DataFrame()
    
    def save_model_performance(self, model_type, metrics, hyperparameters=None):
        """Save model performance metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                hyperparams_json = json.dumps(hyperparameters) if hyperparameters else None
                
                cursor.execute("""
                    INSERT INTO model_performance 
                    (model_type, accuracy, precision_score, recall, f1_score, roc_auc, hyperparameters)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    model_type,
                    metrics.get('accuracy'),
                    metrics.get('precision'),
                    metrics.get('recall'),
                    metrics.get('f1'),
                    metrics.get('roc_auc'),
                    hyperparams_json
                ))
                
                performance_id = cursor.fetchone()[0]
                conn.commit()
                return performance_id
                
        except Exception as e:
            st.error(f"Error saving model performance: {str(e)}")
            return None
    
    def get_model_performance_history(self):
        """Get model performance history"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM model_performance 
                    ORDER BY training_timestamp DESC;
                """
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            st.error(f"Error fetching model performance: {str(e)}")
            return pd.DataFrame()
    
    def save_dataset(self, name, data, description=""):
        """Save uploaded dataset to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                data_json = data.to_json(orient='records')
                file_size = len(data_json)
                num_records = len(data)
                
                cursor.execute("""
                    INSERT INTO datasets 
                    (name, description, data_json, file_size, num_records)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """, (name, description, data_json, file_size, num_records))
                
                dataset_id = cursor.fetchone()[0]
                conn.commit()
                return dataset_id
                
        except Exception as e:
            st.error(f"Error saving dataset: {str(e)}")
            return None
    
    def get_datasets(self):
        """Get list of saved datasets"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT id, name, description, upload_timestamp, file_size, num_records
                    FROM datasets 
                    ORDER BY upload_timestamp DESC;
                """
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            st.error(f"Error fetching datasets: {str(e)}")
            return pd.DataFrame()
    
    def load_dataset(self, dataset_id):
        """Load a specific dataset from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT data_json FROM datasets WHERE id = %s;", (dataset_id,))
                result = cursor.fetchone()
                
                if result:
                    data_json = result[0]
                    df = pd.read_json(data_json, orient='records')
                    return df
                return None
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
    
    def get_statistics(self):
        """Get various statistics from the database"""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Total predictions
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM predictions;")
                stats['total_predictions'] = cursor.fetchone()[0]
                
                # Potable vs non-potable predictions
                cursor.execute("SELECT prediction_label, COUNT(*) FROM predictions GROUP BY prediction_label;")
                prediction_counts = dict(cursor.fetchall())
                stats['potable_predictions'] = prediction_counts.get('Potable', 0)
                stats['non_potable_predictions'] = prediction_counts.get('Not Potable', 0)
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions;")
                avg_confidence = cursor.fetchone()[0]
                stats['average_confidence'] = float(avg_confidence) if avg_confidence else 0
                
                # Most used model
                cursor.execute("""
                    SELECT model_type, COUNT(*) as usage_count 
                    FROM predictions 
                    GROUP BY model_type 
                    ORDER BY usage_count DESC 
                    LIMIT 1;
                """)
                result = cursor.fetchone()
                stats['most_used_model'] = result[0] if result else "None"
                
                # Total datasets
                cursor.execute("SELECT COUNT(*) FROM datasets;")
                stats['total_datasets'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            st.error(f"Error fetching statistics: {str(e)}")
            return {}
    
    def update_session_activity(self, session_id):
        """Update user session activity"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if session exists
                cursor.execute("SELECT id FROM user_sessions WHERE session_id = %s;", (session_id,))
                session = cursor.fetchone()
                
                if session:
                    # Update existing session
                    cursor.execute("""
                        UPDATE user_sessions 
                        SET last_activity = CURRENT_TIMESTAMP 
                        WHERE session_id = %s;
                    """, (session_id,))
                else:
                    # Create new session
                    cursor.execute("""
                        INSERT INTO user_sessions (session_id)
                        VALUES (%s);
                    """, (session_id,))
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Error updating session: {str(e)}")
    
    def export_data(self, table_name, format='csv'):
        """Export data from specified table"""
        try:
            with self.get_connection() as conn:
                if table_name == 'predictions':
                    query = "SELECT * FROM predictions ORDER BY timestamp DESC;"
                elif table_name == 'model_performance':
                    query = "SELECT * FROM model_performance ORDER BY training_timestamp DESC;"
                else:
                    return None
                
                df = pd.read_sql_query(query, conn)
                
                if format == 'csv':
                    return df.to_csv(index=False)
                elif format == 'json':
                    return df.to_json(orient='records', indent=2)
                else:
                    return df
                    
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None