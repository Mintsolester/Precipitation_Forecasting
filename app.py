"""
Streamlit Web App for LSTM Flood Forecasting
Interactive interface for Ganges Basin flood predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train_lstm_flood import LSTMFloodModel

# Configuration
MODEL_PATH = Path(__file__).parent / "runs/lstm_flood_prediction/best_model.pt"
SCALER_PATH = Path(__file__).parent / "runs/lstm_flood_prediction/scaler.pkl"
DATA_FILE = Path(__file__).parent / "data/processed/era5/ganges_farakka_era5_daily.csv"

# Page config
st.set_page_config(
    page_title="Precipitation Forecasting System",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load trained model, scaler, and data"""
    # Load scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMFloodModel(input_size=4, hidden_size=128, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate flood threshold (95th percentile)
    threshold = df['total_precipitation'].quantile(0.95)
    
    return model, scaler, df, threshold, device

def prepare_input_sequence(df, end_date, seq_length=365):
    """Prepare 365-day input sequence for prediction"""
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=seq_length-1)
    
    # Filter data
    sequence_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    if len(sequence_data) < seq_length:
        return None, None, len(sequence_data)
    
    # Select features (must match training: 4 features + target column)
    features = ['total_precipitation', 'temperature_2m', 'soil_moisture_layer1', 'solar_radiation']
    
    # Add target column (same as precipitation for scaler compatibility)
    sequence_data_with_target = sequence_data[features].copy()
    sequence_data_with_target['target_precipitation'] = sequence_data['total_precipitation'].copy()
    
    X = sequence_data_with_target.values
    
    return X, sequence_data, len(sequence_data)

def make_prediction(model, scaler, X, device):
    """Make prediction using the model"""
    # Normalize (X has 5 columns: 4 features + target)
    X_scaled = scaler.transform(X)
    
    # Extract only the 4 features (exclude target column which is last)
    X_features = X_scaled[:, :-1]
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction_scaled = model(X_tensor).cpu().numpy()[0, 0]
    
    # Denormalize (precipitation is first feature)
    mean_precip = scaler.mean_[0]
    std_precip = scaler.scale_[0]
    prediction = prediction_scaled * std_precip + mean_precip
    
    return max(0, prediction)  # Ensure non-negative

# Title
st.markdown('<p class="main-header">�️ Precipitation Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LSTM-based 7-Day Precipitation Forecast for Ganges Basin</p>', unsafe_allow_html=True)

# Load model and data
try:
    with st.spinner('Loading model and data...'):
        model, scaler, df, threshold, device = load_model_and_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Date range info
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    st.sidebar.info(f"""
    **Data Available:**
    - From: {min_date.strftime('%Y-%m-%d')}
    - To: {max_date.strftime('%Y-%m-%d')}
    - Total Days: {len(df)}
    """)
    
    # Prediction date selection
    st.sidebar.subheader("📅 Select Prediction Date")
    
    # Calculate valid prediction range (need 365 days before)
    earliest_prediction = min_date + timedelta(days=364)
    latest_prediction = max_date
    
    prediction_date = st.sidebar.date_input(
        "Prediction Date",
        value=latest_prediction,
        min_value=earliest_prediction,
        max_value=latest_prediction
    )
    
    # Model info
    st.sidebar.subheader("🤖 Model Information")
    st.sidebar.metric("Model Type", "LSTM")
    st.sidebar.metric("F1-Score", "83.3%")
    st.sidebar.metric("Forecast Horizon", "7 days")
    st.sidebar.metric("Input Window", "365 days")
    st.sidebar.metric("Flood Threshold", f"{threshold:.2f} mm/day")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔮 Prediction Results")
        
        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner('Making prediction...'):
                # Prepare input
                X, sequence_data, actual_len = prepare_input_sequence(df, prediction_date)
                
                if X is None:
                    st.error(f"Not enough data! Need 365 days, have {actual_len} days.")
                else:
                    # Make prediction
                    predicted_precip = make_prediction(model, scaler, X, device)
                    
                    # Calculate forecast period
                    forecast_start = prediction_date + timedelta(days=1)
                    forecast_end = forecast_start + timedelta(days=6)
                    
                    # Determine risk level
                    is_high_risk = predicted_precip >= threshold
                    risk_class = "risk-high" if is_high_risk else "risk-low"
                    risk_emoji = "🔴" if is_high_risk else "🟢"
                    risk_text = "HIGH RISK" if is_high_risk else "LOW RISK"
                    
                    # Display results
                    st.markdown(f"""
                    <div class="{risk_class}">
                        {risk_emoji} {risk_text} - {"⚠️ FLOOD WARNING" if is_high_risk else "Normal conditions expected"}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "📈 Predicted MAX Precipitation",
                            f"{predicted_precip:.2f} mm/day",
                            delta=f"{predicted_precip - threshold:.2f} mm/day vs threshold"
                        )
                    
                    with col_b:
                        st.metric(
                            "📅 Forecast Period",
                            f"{forecast_start.strftime('%b %d')} - {forecast_end.strftime('%b %d, %Y')}"
                        )
                    
                    with col_c:
                        risk_percentage = (predicted_precip / threshold * 100) if threshold > 0 else 0
                        st.metric(
                            "⚠️ Risk Level",
                            f"{risk_percentage:.1f}%",
                            f"{'ABOVE' if is_high_risk else 'BELOW'} threshold"
                        )
                    
                    # Recent weather summary
                    st.markdown("---")
                    st.subheader("📊 Recent Weather Conditions (Last 7 Days)")
                    
                    recent_7_days = sequence_data.tail(7)
                    
                    col_d, col_e, col_f, col_g = st.columns(4)
                    
                    with col_d:
                        avg_precip = recent_7_days['total_precipitation'].mean()
                        st.metric("🌧️ Avg Precipitation", f"{avg_precip:.2f} mm/day")
                    
                    with col_e:
                        avg_temp = recent_7_days['temperature_2m'].mean()
                        st.metric("🌡️ Avg Temperature", f"{avg_temp:.2f} °C")
                    
                    with col_f:
                        avg_soil = recent_7_days['soil_moisture_layer1'].mean()
                        st.metric("💧 Avg Soil Moisture", f"{avg_soil:.4f} m³/m³")
                    
                    with col_g:
                        avg_solar = recent_7_days['solar_radiation'].mean()
                        st.metric("☀️ Avg Solar Radiation", f"{avg_solar:.0f} W/m²")
                    
                    # Visualization
                    st.markdown("---")
                    st.subheader("📈 Historical Precipitation Trend (Last 90 Days)")
                    
                    last_90_days = sequence_data.tail(90)
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=last_90_days['date'],
                        y=last_90_days['total_precipitation'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Threshold line
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Flood Threshold ({threshold:.1f} mm/day)",
                        annotation_position="right"
                    )
                    
                    # Prediction point
                    fig.add_trace(go.Scatter(
                        x=[forecast_start + timedelta(days=3)],  # Middle of forecast period
                        y=[predicted_precip],
                        mode='markers',
                        name='Forecast',
                        marker=dict(size=15, color='red' if is_high_risk else 'green', symbol='star')
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Precipitation (mm/day)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ About")
        
        st.info("""
        **How it works:**
        
        1. **Input**: 365 days of weather history
           - Precipitation
           - Temperature
           - Soil moisture
           - Solar radiation
        
        2. **Model**: 2-layer LSTM neural network
        
        3. **Output**: Maximum precipitation expected in next 7 days
        
        4. **Risk Assessment**:
           - 🟢 LOW: < 159 mm/day
           - 🔴 HIGH: ≥ 159 mm/day
        """)
        
        st.warning("""
        ⚠️ **Important Notes:**
        
        - Prediction requires 365 days of prior data
        - Forecast horizon: 7 days ahead
        - Model trained on 2015-2018 data
        - 95th percentile used as flood threshold
        """)
        
        # Download historical data option
        st.subheader("📥 Export Data")
        
        if st.button("Download Historical Data", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ganges_precipitation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌍 NeuralHydrology Precipitation Forecasting System | LSTM-based Precipitation Prediction</p>
        <p>Data source: ERA5-Land Climate Reanalysis | Model: PyTorch LSTM</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading application: {str(e)}")
    st.exception(e)
