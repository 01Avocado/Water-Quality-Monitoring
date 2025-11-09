class WaterQualityDashboard {
    constructor(options = {}) {
        this.dataUrl = options.dataUrl || '../ml_backend/realtime/latest_output.json';
        this.refreshInterval = options.refreshInterval || 60000;
        this.autoRefreshId = null;
        this.lastData = null;
        this.chartCanvas = null;
        this.chartCtx = null;
        this.elements = {};
        this.statusDescriptions = {
            safe: 'Water quality is within recommended limits.',
            degrading: 'Quality is trending downward. Monitor closely.',
            unsafe: 'Water is unsafe for consumption. Immediate action required.'
        };

        this.cacheElements();
        this.init();
    }

    cacheElements() {
        this.elements = {
            statusCard: document.getElementById('waterStatusCard'),
            statusTitle: document.getElementById('waterStatus'),
            statusDescription: document.getElementById('statusDescription'),
            confidence: document.getElementById('confidence'),
            wqiScore: document.getElementById('wqiScore'),
            pollutionLevel: document.getElementById('pollutionLevel'),
            doStatus: document.getElementById('doStatus'),
            lastUpdate: document.getElementById('lastUpdate'),
            connectionStatus: document.getElementById('connectionStatus'),
            connectionStatusText: document.getElementById('connectionStatusText'),
            contaminationCard: document.getElementById('contaminationCard'),
            contaminationType: document.getElementById('contaminationType'),
            contaminationConfidence: document.getElementById('contaminationConfidence'),
            contaminationNote: document.getElementById('contaminationNote'),
            diseaseCard: document.getElementById('diseaseCard'),
            diseasePrediction: document.getElementById('diseasePrediction'),
            diseaseConfidence: document.getElementById('diseaseConfidence'),
            diseaseNote: document.getElementById('diseaseNote'),
            forecastCard: document.getElementById('forecastCard'),
            forecastStatus: document.getElementById('forecastStatus'),
            forecastTimeToUnsafe: document.getElementById('forecastTimeToUnsafe'),
            forecastNote: document.getElementById('forecastNote'),
            alertsSection: document.getElementById('alertsSection'),
            contaminationAlert: document.getElementById('contaminationAlert'),
            alertCause: document.getElementById('contaminationCause'),
            alertDetails: document.getElementById('alertDetails'),
            alertTime: document.getElementById('alertTime'),
            forecastSection: document.getElementById('forecastSection'),
            forecastSummaryText: document.getElementById('forecastSummaryText'),
            forecastTimeline: document.getElementById('forecastTimeline'),
            forecastChartFill: document.getElementById('forecastChartFill'),
            forecastChartLabel: document.getElementById('forecastChartLabel'),
            chartCanvas: document.getElementById('qualityChart'),
            chartEmptyState: document.getElementById('chartEmptyState'),
            modelStatusValue: document.getElementById('modelStatusValue'),
            lastTrainingValue: document.getElementById('lastTrainingValue'),
            overallAccuracyValue: document.getElementById('overallAccuracyValue'),
            monitoringLocation: document.getElementById('monitoringLocation'),
            sensorStatus: document.getElementById('sensorStatus'),
            updateFrequency: document.getElementById('updateFrequency'),
            thingspeakChannel: document.getElementById('thingspeakChannel'),
            infoLastUpdate: document.getElementById('infoLastUpdate'),
            sensorValues: {
                ph: document.getElementById('phValue'),
                tds: document.getElementById('tdsValue'),
                turbidity: document.getElementById('turbidityValue'),
                do: document.getElementById('doValue'),
                temperature: document.getElementById('tempValue'),
            },
            sensorCards: {
                ph: document.getElementById('phValue')?.closest('.metric-card'),
                tds: document.getElementById('tdsValue')?.closest('.metric-card'),
                turbidity: document.getElementById('turbidityValue')?.closest('.metric-card'),
                do: document.getElementById('doValue')?.closest('.metric-card'),
                temperature: document.getElementById('tempValue')?.closest('.metric-card'),
            }
        };
        this.chartCanvas = this.elements.chartCanvas;
        this.chartCtx = this.chartCanvas ? this.chartCanvas.getContext('2d') : null;
    }

    init() {
        this.updateConnectionStatus('connecting');
        this.refreshData();
        this.startAutoRefresh();
    }

    startAutoRefresh() {
        if (this.autoRefreshId) {
            clearInterval(this.autoRefreshId);
        }
        this.autoRefreshId = setInterval(() => this.refreshData(), this.refreshInterval);
    }

    async refreshData() {
        try {
            const response = await fetch(`${this.dataUrl}?v=${Date.now()}`, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.lastData = data;
            this.updateDashboard(data);
            this.updateConnectionStatus('online');
        } catch (error) {
            console.error('Failed to refresh dashboard data:', error);
            this.updateConnectionStatus('offline');
        }
    }

    updateConnectionStatus(state) {
        const statusEl = this.elements.connectionStatus;
        const statusTextEl = this.elements.connectionStatusText;
        if (!statusEl || !statusTextEl) return;

        statusEl.classList.remove('offline');
        switch (state) {
            case 'online':
                statusTextEl.textContent = 'Connected';
                statusEl.querySelector('i').style.color = 'var(--success-color)';
                break;
            case 'offline':
                statusEl.classList.add('offline');
                statusTextEl.textContent = 'Offline';
                statusEl.querySelector('i').style.color = 'var(--danger-color)';
                break;
            default:
                statusTextEl.textContent = 'Connecting...';
                statusEl.querySelector('i').style.color = 'var(--warning-color)';
        }
    }

    updateDashboard(data) {
        if (!data) return;

        this.updateTimestamp(data.timestamp);
        this.updateStatusCard(data);
        this.updateSensorCards(data.sensor_values || {});
        this.updateContamination(data.contamination || {});
        this.updateDisease(data.disease || {});
        this.updateForecast(data.degradation || {}, data.wqi || {});
        this.updateAlerts(data);
        this.updateSystemInfo(data);
        this.updateChart(data.degradation || {});
    }

    updateTimestamp(timestamp) {
        const formatted = timestamp ? this.formatTimestamp(timestamp) : 'Unavailable';
        if (this.elements.lastUpdate) {
            this.elements.lastUpdate.textContent = formatted;
        }
        if (this.elements.infoLastUpdate) {
            this.elements.infoLastUpdate.textContent = formatted;
        }
    }

    updateStatusCard(data) {
        const wqi = data.wqi || {};
        const sensorValues = data.sensor_values || {};
        const status = (wqi.status || 'Safe').toLowerCase();
        const confidence = this.getWqiConfidence(wqi);

        if (this.elements.statusCard) {
            this.elements.statusCard.classList.remove('safe', 'degrading', 'unsafe', 'changing');
            this.elements.statusCard.classList.add(status);
            this.elements.statusCard.classList.add('changing');
            setTimeout(() => this.elements.statusCard.classList.remove('changing'), 400);
        }

        if (this.elements.statusTitle) {
            this.elements.statusTitle.textContent = wqi.status || 'Water Status';
        }
        if (this.elements.statusDescription) {
            this.elements.statusDescription.textContent = this.statusDescriptions[status] || '';
        }
        if (this.elements.confidence && confidence !== null) {
            this.elements.confidence.textContent = confidence;
        }
        if (this.elements.wqiScore && wqi.wqi_score != null) {
            this.elements.wqiScore.textContent = wqi.wqi_score.toFixed(1);
        }
        if (this.elements.pollutionLevel) {
            const levelMap = { 0: 'Safe', 1: 'Degrading', 2: 'Unsafe' };
            this.elements.pollutionLevel.textContent = levelMap[wqi.pollution_level] || wqi.status || 'Unknown';
        }
        if (this.elements.doStatus) {
            this.elements.doStatus.textContent = sensorValues.DO_imputed ? 'Imputed' : 'Live';
        }
    }

    updateSensorCards(values) {
        const defaults = { ph: null, TDS: null, Turbidity: null, DO: null, Temperature: null };
        const readings = { ...defaults, ...this.normalizeSensorKeys(values) };

        Object.entries(readings).forEach(([key, value]) => {
            const lowerKey = key.toLowerCase();
            const element = this.elements.sensorValues[lowerKey];
            if (element && value != null) {
                element.textContent = this.formatNumber(value, lowerKey === 'tds' ? 0 : 1);
            }
        });

        this.updateMetricStatus('ph', readings.pH, [6.5, 8.5]);
        this.updateMetricStatus('tds', readings.TDS, [50, 500]);
        this.updateMetricStatus('turbidity', readings.Turbidity, [0, 5]);
        this.updateMetricStatus('do', readings.DO, [6, 10], values.DO_imputed);
        this.updateMetricStatus('temperature', readings.Temperature, [15, 30]);
    }

    normalizeSensorKeys(values) {
        return {
            pH: values.pH ?? values.ph ?? null,
            TDS: values.TDS ?? values.tds_mg_l ?? values.tds ?? null,
            Turbidity: values.Turbidity ?? values.turbidity ?? values.turbidity_ntu ?? null,
            DO: values.DO ?? values.do_mg_l ?? values.do ?? null,
            Temperature: values.Temperature ?? values.temperature ?? values.temperature_c ?? null,
            DO_imputed: values.DO_imputed,
        };
    }

    updateMetricStatus(key, value, range, imputed = false) {
        const card = this.elements.sensorCards[key];
        if (!card) return;

        const statusEl = card.querySelector('.metric-status');
        if (!statusEl) return;

        statusEl.classList.remove('good', 'warning', 'danger');

        if (imputed && key === 'do') {
            statusEl.classList.add('warning');
            statusEl.textContent = 'Imputed';
            return;
        }

        if (value == null) {
            statusEl.classList.add('warning');
            statusEl.textContent = 'No Data';
            return;
        }

        const [min, max] = range;
        if (value >= min && value <= max) {
            statusEl.classList.add('good');
            statusEl.textContent = 'Normal';
        } else if (value < min * 0.8 || value > max * 1.2) {
            statusEl.classList.add('danger');
            statusEl.textContent = 'Critical';
        } else {
            statusEl.classList.add('warning');
            statusEl.textContent = 'Warning';
        }
    }

    updateContamination(contamination) {
        if (!this.elements.contaminationCard) return;

        const type = contamination.contamination_type || 'safe';
        const confidence = contamination.confidence ?? this.getMaxProbability(contamination.all_probabilities);

        this.elements.contaminationCard.classList.remove('safe', 'warning', 'alert');
        const cardState = type === 'safe' ? 'safe' : 'alert';
        this.elements.contaminationCard.classList.add(cardState);

        this.elements.contaminationType.textContent = this.formatContaminationType(type);
        this.elements.contaminationConfidence.textContent = this.formatConfidence(confidence);

        if (type === 'safe') {
            this.elements.contaminationNote.textContent = 'No contamination detected.';
        } else {
            this.elements.contaminationNote.textContent = contamination.description ||
                'Contamination detected. Follow recommended mitigation steps.';
        }
    }

    updateDisease(disease) {
        if (!this.elements.diseaseCard) return;

        const prediction = disease.predicted_disease || 'No Disease';
        const confidence = disease.confidence ?? this.getMaxProbability(disease.all_probabilities);
        const severity = (disease.severity || 'SAFE').toUpperCase();

        this.elements.diseaseCard.classList.remove('safe', 'warning', 'alert');
        if (severity === 'CRITICAL' || severity === 'HIGH') {
            this.elements.diseaseCard.classList.add('alert');
        } else if (severity === 'MEDIUM') {
            this.elements.diseaseCard.classList.add('warning');
        } else {
            this.elements.diseaseCard.classList.add('safe');
        }

        this.elements.diseasePrediction.textContent = prediction;
        this.elements.diseaseConfidence.textContent = this.formatConfidence(confidence);
        this.elements.diseaseNote.textContent = disease.health_warning ||
            'Waterborne disease risk is minimal.';
    }

    updateForecast(degradation, wqi) {
        const timeline = degradation.forecast_timeline || [];
        const timeToUnsafe = degradation.time_to_unsafe || {};
        const availability = degradation.status;

        this.elements.forecastCard.classList.remove('safe', 'warning', 'alert');

        if (availability === 'unavailable') {
            this.elements.forecastCard.classList.add('warning');
            this.elements.forecastStatus.textContent = 'Forecast unavailable';
            this.elements.forecastTimeToUnsafe.textContent = '--';
            this.elements.forecastNote.textContent = degradation.message ||
                'Install TensorFlow on the backend to enable forecasting.';
            this.toggleForecastSection(false);
            return;
        }

        if (timeline.length === 0) {
            this.elements.forecastCard.classList.add('warning');
            this.elements.forecastStatus.textContent = 'Awaiting history';
            this.elements.forecastTimeToUnsafe.textContent = '--';
            this.elements.forecastNote.textContent = degradation.message ||
                'The system needs more recent readings before forecasting.';
            this.toggleForecastSection(false);
            return;
        }

        const isUnsafeSoon = timeToUnsafe.will_become_unsafe;
        if (isUnsafeSoon) {
            this.elements.forecastCard.classList.add('alert');
            this.elements.forecastStatus.textContent = 'Unsafe risk detected';
            this.elements.forecastTimeToUnsafe.textContent =
                `${timeToUnsafe.hours_to_unsafe?.toFixed(1) ?? '--'} hrs`;
            this.elements.forecastNote.textContent =
                `Predicted unsafe WQI: ${timeToUnsafe.predicted_unsafe_wqi?.toFixed(1) ?? '--'}`;
        } else {
            this.elements.forecastCard.classList.add('safe');
            this.elements.forecastStatus.textContent = 'Stable';
            this.elements.forecastTimeToUnsafe.textContent = 'Not in forecast window';
            this.elements.forecastNote.textContent =
                `Min predicted WQI: ${degradation.min_predicted_wqi?.toFixed(1) ?? '--'}`;
        }

        this.elements.forecastSummaryText.textContent =
            `Forecast horizon: ${degradation.forecast_horizon_hours || wqi.forecast_horizon_hours || 12} hours`;

        this.renderForecastTimeline(timeline);

        if (timeline.length) {
            const minValue = Math.min(...timeline.map(item => item.predicted_wqi));
            const fillPercent = Math.max(0, Math.min(100, (minValue / 100) * 100));
            if (this.elements.forecastChartFill) {
                this.elements.forecastChartFill.style.width = `${fillPercent}%`;
            }
            if (this.elements.forecastChartLabel) {
                this.elements.forecastChartLabel.textContent = `Lowest predicted WQI: ${minValue.toFixed(1)}`;
            }
        }

        this.toggleForecastSection(true);
    }

    renderForecastTimeline(timeline) {
        if (!this.elements.forecastTimeline) return;
        this.elements.forecastTimeline.innerHTML = '';

        timeline.forEach((item, index) => {
            const timelineItem = document.createElement('div');
            timelineItem.className = 'timeline-item';

            const marker = document.createElement('div');
            marker.className = 'timeline-marker';
            marker.classList.add(index === 0 ? 'current' : (item.status === 'Unsafe' ? 'future' : 'current'));

            const content = document.createElement('div');
            content.className = 'timeline-content';

            const title = document.createElement('h4');
            title.textContent = index === 0 ? 'Current Status' : `${item.hours_ahead}h Forecast`;

            const description = document.createElement('p');
            description.textContent = `${item.status} (WQI ${item.predicted_wqi.toFixed(1)})`;

            content.appendChild(title);
            content.appendChild(description);

            timelineItem.appendChild(marker);
            timelineItem.appendChild(content);
            this.elements.forecastTimeline.appendChild(timelineItem);
        });
    }

    toggleForecastSection(visible) {
        if (!this.elements.forecastSection) return;
        this.elements.forecastSection.style.display = visible ? 'block' : 'none';
    }

    updateAlerts(data) {
        const contamination = data.contamination || {};
        const disease = data.disease || {};
        const timestamp = data.timestamp;

        const shouldAlert =
            (contamination.contamination_type && contamination.contamination_type !== 'safe') ||
            (disease.severity && ['HIGH', 'CRITICAL'].includes(disease.severity.toUpperCase()));

        if (!this.elements.alertsSection) return;

        if (!shouldAlert) {
            this.elements.alertsSection.style.display = 'none';
            return;
        }

        this.elements.alertsSection.style.display = 'block';
        if (this.elements.alertCause) {
            if (contamination.contamination_type && contamination.contamination_type !== 'safe') {
                this.elements.alertCause.textContent = `Contamination: ${this.formatContaminationType(contamination.contamination_type)}`;
            } else {
                this.elements.alertCause.textContent = `Disease Risk: ${disease.predicted_disease || 'High'}`;
            }
        }
        if (this.elements.alertDetails) {
            this.elements.alertDetails.textContent = contamination.description ||
                disease.health_warning ||
                'Take immediate mitigation steps.';
        }
        if (this.elements.alertTime) {
            this.elements.alertTime.textContent = this.formatRelativeTime(timestamp);
        }
    }

    updateSystemInfo(data) {
        if (this.elements.modelStatusValue) {
            this.elements.modelStatusValue.textContent = data.degradation?.status === 'unavailable'
                ? 'Partial (Forecast disabled)'
                : 'Active';
            this.elements.modelStatusValue.classList.toggle(
                'active',
                data.degradation?.status !== 'unavailable'
            );
        }
        if (this.elements.lastTrainingValue) {
            this.elements.lastTrainingValue.textContent = 'Refer to training pipeline logs';
        }
        if (this.elements.overallAccuracyValue) {
            const wqiConfidence = this.getMaxProbability(data.wqi?.probabilities);
            this.elements.overallAccuracyValue.textContent = wqiConfidence
                ? `${wqiConfidence}`
                : '--';
        }
        if (this.elements.monitoringLocation) {
            this.elements.monitoringLocation.textContent =
                data.raw_channel?.name || 'ThingSpeak Channel';
        }
        if (this.elements.sensorStatus) {
            this.elements.sensorStatus.textContent = data.sensor_values?.DO_imputed
                ? '4 Live + DO (imputed)'
                : '5 Active Sensors';
        }
        if (this.elements.updateFrequency) {
            this.elements.updateFrequency.textContent = 'Configured in realtime pipeline';
        }
        if (this.elements.thingspeakChannel) {
            this.elements.thingspeakChannel.textContent =
                data.raw_channel?.id ? `ID ${data.raw_channel.id}` : 'Unknown';
        }
    }

    updateChart(degradation) {
        if (!this.chartCanvas || !this.chartCtx) return;

        const timeline = degradation.forecast_timeline || [];
        if (!timeline.length) {
            this.setChartEmpty(true, degradation.message || 'Forecast data not available yet.');
            return;
        }

        this.setChartEmpty(false);
        const ctx = this.chartCtx;
        const width = this.chartCanvas.width;
        const height = this.chartCanvas.height;
        ctx.clearRect(0, 0, width, height);

        const padding = 50;
        const chartWidth = width - (padding * 2);
        const chartHeight = height - (padding * 2);

        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        ctx.strokeStyle = '#f1f5f9';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = padding + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.beginPath();
        timeline.forEach((point, index) => {
            const x = padding + (chartWidth / (timeline.length - 1 || 1)) * index;
            const y = height - padding - (Math.min(point.predicted_wqi, 100) / 100) * chartHeight;
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        ctx.fillStyle = '#2563eb';
        timeline.forEach((point, index) => {
            const x = padding + (chartWidth / (timeline.length - 1 || 1)) * index;
            const y = height - padding - (Math.min(point.predicted_wqi, 100) / 100) * chartHeight;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        ctx.fillStyle = '#64748b';
        ctx.font = '12px Inter';
        ctx.textAlign = 'right';
        [0, 25, 50, 75, 100].forEach((value, index) => {
            const y = height - padding - (chartHeight / 100) * value;
            ctx.fillText(`${value}`, padding - 8, y + 4);
        });

        ctx.textAlign = 'center';
        timeline.forEach((point, index) => {
            const x = padding + (chartWidth / (timeline.length - 1 || 1)) * index;
            const label = index === 0 ? 'Now' : `+${point.hours_ahead}h`;
            ctx.fillText(label, x, height - padding + 20);
        });

        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 16px Inter';
        ctx.fillText('Forecasted WQI Trend', width / 2, 30);
    }

    setChartEmpty(show, message = '') {
        if (!this.elements.chartEmptyState) return;
        this.elements.chartEmptyState.style.display = show ? 'block' : 'none';
        if (message) {
            this.elements.chartEmptyState.textContent = message;
        }
        if (this.chartCanvas) {
            this.chartCanvas.style.opacity = show ? '0.3' : '1';
        }
    }

    getWqiConfidence(wqi) {
        if (!wqi || !wqi.probabilities) return null;
        const label = wqi.status || 'Safe';
        const prob = wqi.probabilities[label] ?? this.getMaxProbability(wqi.probabilities, true);
        return this.formatConfidence(prob);
    }

    getMaxProbability(probabilities, raw = false) {
        if (!probabilities) return raw ? null : '--';
        const values = Object.values(probabilities);
        if (!values.length) return raw ? null : '--';
        const max = Math.max(...values);
        return raw ? max : this.formatConfidence(max);
    }

    formatConfidence(value) {
        if (value == null) return '--';
        return `${(value * 100).toFixed(1)}%`;
    }

    formatContaminationType(type) {
        if (!type) return 'Unknown';
        const formatted = type.replace(/_/g, ' ');
        return formatted.charAt(0).toUpperCase() + formatted.slice(1);
    }

    formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            if (Number.isNaN(date.getTime())) return 'Invalid';
            return date.toLocaleString();
        } catch {
            return 'Invalid';
        }
    }

    formatRelativeTime(timestamp) {
        if (!timestamp) return 'Unknown';
        const eventTime = new Date(timestamp);
        const now = new Date();
        const diffMs = now - eventTime;
        if (Number.isNaN(diffMs)) return 'Unknown';
        const diffMinutes = Math.round(diffMs / 60000);
        if (diffMinutes <= 1) return 'Just now';
        if (diffMinutes < 60) return `${diffMinutes} min ago`;
        const diffHours = Math.round(diffMinutes / 60);
        return `${diffHours} hr ago`;
    }

    formatNumber(value, decimals = 1) {
        if (value == null || Number.isNaN(value)) return '--';
        return Number(value).toFixed(decimals);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new WaterQualityDashboard({
        dataUrl: window.DASHBOARD_DATA_URL || '../ml_backend/realtime/latest_output.json',
        refreshInterval: window.DASHBOARD_REFRESH_INTERVAL || 60_000,
    });
    window.waterQualityDashboard = dashboard;
});
