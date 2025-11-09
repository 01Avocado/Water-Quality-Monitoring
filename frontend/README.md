# Water Quality Monitoring Dashboard

A modern, responsive web dashboard for real-time water quality monitoring using AI-powered contamination detection.

## Features

### 🎯 Core Functionality
- **Water Safety Status**: Real-time display powered by the WQI classifier
- **Contamination Watch**: Live predictions from the contamination detection model
- **Disease Risk**: Outbreak risk assessment with severity guidance
- **Degradation Outlook**: Forecast timeline driven by the LSTM degradation model (auto-disables if TensorFlow unavailable)
- **Sensor Monitoring**: Live sensor grid with automatic DO imputation flag

### 🎨 Design Features
- **Modern UI**: Clean, professional design with intuitive user interface
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Real-time Updates**: Dynamic data visualization with smooth animations
- **Status Indicators**: Color-coded status system (Green: Safe, Yellow: Degrading, Red: Unsafe)

### 📊 Data Visualization
- **Sensor Metrics Grid**: Individual cards for each water quality parameter
- **Historical Trends**: 24-hour quality trend chart
- **System Information**: Model status, accuracy, and monitoring details
- **Interactive Elements**: Hover effects and smooth transitions

## File Structure

```
frontend/
├── index.html          # Main HTML structure
├── styles.css          # Complete CSS styling with responsive design
├── script.js           # JavaScript for dynamic functionality and demo
└── README.md           # This documentation file
```

## Getting Started

1. **Run the realtime pipeline** (in a separate shell):
   ```bash
   export THINGSPEAK_CHANNEL_ID=YOUR_CHANNEL_ID
   export THINGSPEAK_READ_API_KEY=YOUR_READ_KEY        # optional if channel is public
   python -m ml_backend.realtime.run_service --loop
   ```
2. **Serve the frontend** (from the repository root):
   ```bash
   python -m http.server 8000
   ```
3. **Open the dashboard** at `http://localhost:8000/frontend/index.html`

The dashboard polls `../ml_backend/realtime/latest_output.json` every 60 seconds. You can override the data source or refresh cadence by setting:

```html
<script>
  window.DASHBOARD_DATA_URL = '/custom/path/latest_output.json';
  window.DASHBOARD_REFRESH_INTERVAL = 30000; // 30 seconds
</script>
```

## Dashboard Components

### 1. Header
- System logo and title
- Connection status indicator
- Last update timestamp

### 2. Water Status Overview
- Large status card showing current water safety
- Confidence percentage
- Color-coded status (Safe/Degrading/Unsafe)

### 3. Alert System
- **Contamination Alerts**: Shows when water is contaminated with cause details
- **Degradation Forecast**: Timeline showing predicted contamination time
- Action buttons for alerts

### 4. Sensor Metrics
- Live sensor readings with dynamic status badges
- DO card highlights imputed values when the physical probe is offline

### 5. Forecast Chart
- Forecasted WQI trend plotted from the degradation model output
- Graceful fallbacks when TensorFlow is not installed or insufficient history is available

### 6. System Information
- Model health, ThingSpeak channel, sensor mix, last refresh timestamp

## Color Scheme

- **Safe (Green)**: `#10b981` - Water is safe for consumption
- **Degrading (Yellow)**: `#f59e0b` - Quality is declining but still safe
- **Unsafe (Red)**: `#ef4444` - Water is contaminated and unsafe
- **Primary Blue**: `#2563eb` - Main brand color
- **Neutral Grays**: Various shades for text and backgrounds

## Responsive Breakpoints

- **Desktop**: 1200px+ (Full layout with all features)
- **Tablet**: 768px - 1199px (Adjusted grid layouts)
- **Mobile**: < 768px (Single column, stacked layout)

## Future Integration

The dashboard is designed to easily integrate with your LSTM model:

### JavaScript API Structure
```javascript
// Fetch real sensor data
const sensorData = await WaterQualityAPI.fetchSensorData();

// Get AI predictions
const prediction = await WaterQualityAPI.predictContamination(sensorData);

// Get historical data
const history = await WaterQualityAPI.getHistoricalData(24);
```

### Backend Integration Points
1. **Real-time Data**: Replace demo data with actual sensor readings
2. **AI Predictions**: Connect to your LSTM model for contamination detection
3. **Alert System**: Integrate with notification services
4. **Historical Data**: Connect to your database for trend analysis

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Performance Features

- **Optimized CSS**: Efficient styling with CSS custom properties
- **Smooth Animations**: Hardware-accelerated transitions
- **Responsive Images**: Optimized for different screen sizes
- **Fast Loading**: Minimal dependencies, only Font Awesome icons

## Customization

### Colors
Modify the CSS custom properties in `styles.css`:
```css
:root {
    --primary-color: #2563eb;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
}
```

### Layout
Adjust grid layouts and spacing:
```css
.metrics-grid {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
}
```

## Live Integration Notes

- The dashboard fetches the same JSON artefact produced by the realtime pipeline (`ml_backend/realtime/latest_output.json`). Host both assets from the same domain to avoid CORS issues.
- When TensorFlow is missing on the backend, the dashboard clearly indicates that the degradation forecast is unavailable.
- DO readings display an "Imputed" status whenever the machine-learning proxy supplies the value.

## Support

This dashboard is designed to be easily maintainable and extensible. The modular structure allows for easy updates and feature additions as your water quality monitoring system evolves.
