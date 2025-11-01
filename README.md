# 🌧️ Precipitation Forecasting System

LSTM-based 7-day precipitation forecast for the Ganges Basin using ERA5-Land climate reanalysis data.

## 🌟 Features

- **Interactive Web Interface** - User-friendly Streamlit dashboard
- **7-Day Forecasts** - Predict maximum precipitation for the next week
- **Flood Risk Assessment** - Automatic HIGH/LOW risk classification
- **Historical Visualization** - 90-day precipitation trends with forecast overlay
- **Weather Metrics** - Recent conditions summary (temperature, soil moisture, solar radiation)
- **Data Export** - Download historical data as CSV

## 📊 Model Performance

- **F1-Score**: 83.3%
- **RMSE**: 24.02 mm/day
- **Architecture**: 2-layer LSTM (128 hidden units, 209K parameters)
- **Training Data**: 2015-2018 ERA5-Land reanalysis
- **Input Window**: 365 days of weather history
- **Forecast Horizon**: 7 days ahead

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/neuralhydrology-flood-forecast.git
cd neuralhydrology-flood-forecast

# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Docker

```bash
# Build image
docker build -t precipitation-forecast .

# Run container
docker run -p 8501:8501 precipitation-forecast
```

## 📝 Usage

1. **Select a prediction date** from the sidebar calendar
2. **Click "Generate Forecast"** to create prediction
3. **View results**:
   - 🟢 **LOW RISK**: Precipitation below flood threshold (159 mm/day)
   - 🔴 **HIGH RISK**: Precipitation at or above flood threshold
4. **Explore visualizations**:
   - Historical 90-day precipitation trend
   - Forecast point overlay
   - Recent weather conditions

## 📂 Data

### Input Features:
- **Total Precipitation** (mm/day)
- **Temperature** (°C)
- **Soil Moisture** (m³/m³)
- **Solar Radiation** (W/m²)

### Data Source:
ERA5-Land hourly climate reanalysis from Copernicus CDS

### Coverage:
- **Spatial**: Ganges Basin (73-89°E, 22-31°N)
- **Temporal**: 2015-2018 (training), 2024-2025 (operational)
- **Resolution**: Daily aggregates from hourly data

## 🏗️ Technical Architecture

```
Input (365 days × 4 features)
    ↓
[LSTM Layer 1] (128 units)
    ↓
[LSTM Layer 2] (128 units)
    ↓
[Fully Connected] (64 units)
    ↓
[ReLU + Dropout]
    ↓
[Output] (1 value: max precipitation in next 7 days)
```

## 📁 Project Structure

```
neuralhydrology-flood-forecast/
├── app.py                          # Streamlit web application
├── scripts/
│   ├── train_lstm_flood.py        # Model training script
│   ├── predict_flood.py           # CLI prediction tool
│   └── preprocessing/             # Data processing scripts
├── runs/
│   └── lstm_flood_prediction/
│       ├── best_model.pt          # Trained model weights
│       └── scaler.pkl             # Feature scaler
├── data/
│   └── processed/
│       └── era5/
│           └── ganges_farakka_era5_daily.csv
└── requirements_streamlit.txt     # Python dependencies
```

## 🔧 Configuration

### Flood Threshold
- **Current**: 159.13 mm/day (95th percentile)
- **Adjustable** in code or via percentile calculation

### Model Parameters
- **Sequence Length**: 365 days
- **Forecast Horizon**: 7 days
- **Hidden Size**: 128
- **Layers**: 2
- **Dropout**: 0.3

## 📈 Metrics Explained

### Regression Metrics:
- **RMSE**: Root Mean Squared Error in mm/day
- **MAE**: Mean Absolute Error in mm/day

### Classification Metrics:
- **Precision**: Accuracy of flood warnings (reduce false alarms)
- **Recall**: Ability to detect all flood events (minimize misses)
- **F1-Score**: Balance between precision and recall

## 🌍 Use Cases

- **Flood Early Warning** - 7-day advance notice for emergency preparedness
- **Agricultural Planning** - Irrigation scheduling based on rainfall forecasts
- **Water Resource Management** - Reservoir operations and flood control
- **Climate Research** - Extreme precipitation pattern analysis

## ⚠️ Limitations

- Requires **365 consecutive days** of prior data for prediction
- Model trained on **2015-2018** - performance may vary for different climate regimes
- Predicts **maximum** precipitation in 7-day window (not daily values)
- **Regional**: Optimized for Ganges Basin - may need retraining for other regions

## 🔮 Future Enhancements

- [ ] Daily precipitation forecasts (instead of 7-day maximum)
- [ ] Multi-basin support
- [ ] Ensemble forecasting with uncertainty quantification
- [ ] Real-time data integration with CDS API
- [ ] Mobile-responsive design
- [ ] Email/SMS alert system

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

**Author**: Your Name  
**Email**: your.email@example.com  
**GitHub**: https://github.com/YOUR_USERNAME

## 🙏 Acknowledgments

- **ERA5-Land Data**: Copernicus Climate Change Service (C3S)
- **NeuralHydrology Framework**: Kratzert et al.
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.

---

**Built with ❤️ for flood resilience**
