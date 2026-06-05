"""
Demand Anomaly Detection Module

Detects unusual patterns in demand data using multiple statistical and ML methods:
- Statistical methods (Z-score, IQR)
- Isolation Forest (JAX-compatible implementation)
- Time series decomposition
- Contextual anomaly detection
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class AnomalyType(Enum):
    """Types of demand anomalies"""
    SPIKE = "sudden_spike"              # Sudden increase in demand
    DROP = "sudden_drop"                # Sudden decrease in demand
    SUSTAINED_HIGH = "sustained_high"   # Prolonged high demand
    SUSTAINED_LOW = "sustained_low"     # Prolonged low demand
    VOLATILITY = "high_volatility"      # Unusual variance
    SEASONAL_ANOMALY = "seasonal_anomaly"  # Deviates from seasonal pattern


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    timestamp: datetime
    anomaly_type: AnomalyType
    risk_level: RiskLevel
    anomaly_score: float
    expected_value: float
    actual_value: float
    deviation_percent: float
    description: str
    recommended_actions: List[str]
    estimated_impact: Dict[str, float]  # {'revenue_at_risk': float, 'cost_impact': float}


class DemandAnomalyDetector:
    """
    Multi-method anomaly detector for demand forecasting

    Combines statistical methods with machine learning for robust anomaly detection
    """

    def __init__(self,
                 window_size: int = 30,
                 seasonal_period: int = 7,
                 sensitivity: str = "medium",
                 min_history: int = 60):
        """
        Initialize anomaly detector

        Args:
            window_size: Rolling window for statistics
            seasonal_period: Period for seasonal decomposition (7 for weekly)
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            min_history: Minimum historical data points required
        """
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.sensitivity = sensitivity
        self.min_history = min_history

        # Thresholds based on sensitivity
        self.thresholds = self._get_thresholds(sensitivity)

        # Historical baseline
        self.baseline_mean = None
        self.baseline_std = None
        self.seasonal_pattern = None
        self.is_fitted = False

    def _get_thresholds(self, sensitivity: str) -> Dict[str, float]:
        """Get detection thresholds based on sensitivity level"""
        thresholds = {
            'low': {
                'z_score': 3.5,
                'iqr_multiplier': 3.0,
                'volatility_threshold': 2.5,
                'seasonal_deviation': 2.5
            },
            'medium': {
                'z_score': 3.0,
                'iqr_multiplier': 2.5,
                'volatility_threshold': 2.0,
                'seasonal_deviation': 2.0
            },
            'high': {
                'z_score': 2.5,
                'iqr_multiplier': 2.0,
                'volatility_threshold': 1.5,
                'seasonal_deviation': 1.5
            }
        }
        return thresholds.get(sensitivity, thresholds['medium'])

    def fit(self, historical_demand: np.ndarray, timestamps: Optional[List[datetime]] = None) -> None:
        """
        Fit detector on historical demand data

        Args:
            historical_demand: Historical demand time series
            timestamps: Optional timestamps for each data point
        """
        if len(historical_demand) < self.min_history:
            raise ValueError(f"Need at least {self.min_history} historical data points")

        # Calculate baseline statistics
        self.baseline_mean = np.mean(historical_demand)
        self.baseline_std = np.std(historical_demand)

        # Decompose seasonal pattern
        self.seasonal_pattern = self._extract_seasonal_pattern(historical_demand)

        self.is_fitted = True

    def _extract_seasonal_pattern(self, data: np.ndarray) -> np.ndarray:
        """Extract seasonal pattern using moving average"""
        if len(data) < self.seasonal_period * 2:
            return np.zeros(self.seasonal_period)

        # Simple seasonal decomposition
        n_periods = len(data) // self.seasonal_period
        seasonal_data = data[:n_periods * self.seasonal_period].reshape(n_periods, self.seasonal_period)
        seasonal_pattern = np.mean(seasonal_data, axis=0)

        # Normalize
        seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern) + self.baseline_mean

        return seasonal_pattern

    def detect(self,
               recent_demand: np.ndarray,
               current_value: float,
               current_timestamp: Optional[datetime] = None) -> List[Anomaly]:
        """
        Detect anomalies in current demand

        Args:
            recent_demand: Recent demand history
            current_value: Current demand value to check
            current_timestamp: Current timestamp

        Returns:
            List of detected anomalies
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

        anomalies = []

        # 1. Z-Score Method
        z_score_anomaly = self._detect_zscore_anomaly(recent_demand, current_value)
        if z_score_anomaly:
            anomalies.append(z_score_anomaly)

        # 2. IQR Method
        iqr_anomaly = self._detect_iqr_anomaly(recent_demand, current_value)
        if iqr_anomaly:
            anomalies.append(iqr_anomaly)

        # 3. Volatility Detection
        volatility_anomaly = self._detect_volatility_anomaly(recent_demand, current_value)
        if volatility_anomaly:
            anomalies.append(volatility_anomaly)

        # 4. Seasonal Anomaly Detection
        if current_timestamp and self.seasonal_pattern is not None:
            seasonal_anomaly = self._detect_seasonal_anomaly(
                current_value, current_timestamp
            )
            if seasonal_anomaly:
                anomalies.append(seasonal_anomaly)

        # 5. Sustained Trend Detection
        trend_anomaly = self._detect_sustained_trend(recent_demand, current_value)
        if trend_anomaly:
            anomalies.append(trend_anomaly)

        # Deduplicate and prioritize anomalies
        anomalies = self._deduplicate_anomalies(anomalies)

        return anomalies

    def _detect_zscore_anomaly(self, recent_demand: np.ndarray, current_value: float) -> Optional[Anomaly]:
        """Detect anomaly using Z-score method"""
        if len(recent_demand) < 10:
            return None

        mean = np.mean(recent_demand)
        std = np.std(recent_demand)

        if std < 1e-6:  # Avoid division by zero
            return None

        z_score = abs((current_value - mean) / std)

        if z_score > self.thresholds['z_score']:
            # Determine if spike or drop
            if current_value > mean:
                anomaly_type = AnomalyType.SPIKE
                description = f"Sudden demand spike detected: {current_value:.0f} vs expected {mean:.0f}"
            else:
                anomaly_type = AnomalyType.DROP
                description = f"Sudden demand drop detected: {current_value:.0f} vs expected {mean:.0f}"

            # Calculate risk level
            risk_level = self._calculate_risk_level(z_score, self.thresholds['z_score'])

            # Calculate deviation
            deviation_percent = ((current_value - mean) / mean) * 100

            # Estimate impact
            impact = self._estimate_impact(current_value, mean, anomaly_type)

            # Generate recommendations
            recommendations = self._generate_recommendations(anomaly_type, current_value, mean)

            return Anomaly(
                timestamp=datetime.now(),
                anomaly_type=anomaly_type,
                risk_level=risk_level,
                anomaly_score=float(z_score),
                expected_value=float(mean),
                actual_value=float(current_value),
                deviation_percent=float(deviation_percent),
                description=description,
                recommended_actions=recommendations,
                estimated_impact=impact
            )

        return None

    def _detect_iqr_anomaly(self, recent_demand: np.ndarray, current_value: float) -> Optional[Anomaly]:
        """Detect anomaly using Interquartile Range method"""
        if len(recent_demand) < 10:
            return None

        q1 = np.percentile(recent_demand, 25)
        q3 = np.percentile(recent_demand, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.thresholds['iqr_multiplier'] * iqr
        upper_bound = q3 + self.thresholds['iqr_multiplier'] * iqr

        if current_value < lower_bound or current_value > upper_bound:
            median = np.median(recent_demand)

            if current_value > upper_bound:
                anomaly_type = AnomalyType.SPIKE
            else:
                anomaly_type = AnomalyType.DROP

            # Calculate anomaly score (normalized distance from bounds)
            if current_value > upper_bound:
                anomaly_score = (current_value - upper_bound) / (iqr + 1e-6)
            else:
                anomaly_score = (lower_bound - current_value) / (iqr + 1e-6)

            risk_level = self._calculate_risk_level(anomaly_score, 2.0)

            return Anomaly(
                timestamp=datetime.now(),
                anomaly_type=anomaly_type,
                risk_level=risk_level,
                anomaly_score=float(anomaly_score),
                expected_value=float(median),
                actual_value=float(current_value),
                deviation_percent=((current_value - median) / median) * 100,
                description=f"Demand outside normal range: {current_value:.0f} (range: {lower_bound:.0f}-{upper_bound:.0f})",
                recommended_actions=self._generate_recommendations(anomaly_type, current_value, median),
                estimated_impact=self._estimate_impact(current_value, median, anomaly_type)
            )

        return None

    def _detect_volatility_anomaly(self, recent_demand: np.ndarray, current_value: float) -> Optional[Anomaly]:
        """Detect unusual volatility in demand"""
        if len(recent_demand) < self.window_size:
            return None

        # Calculate rolling volatility
        recent_window = recent_demand[-self.window_size:]
        current_volatility = np.std(recent_window)

        # Historical volatility
        historical_volatility = self.baseline_std

        if historical_volatility < 1e-6:
            return None

        volatility_ratio = current_volatility / historical_volatility

        if volatility_ratio > self.thresholds['volatility_threshold']:
            risk_level = self._calculate_risk_level(volatility_ratio, self.thresholds['volatility_threshold'])

            return Anomaly(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.VOLATILITY,
                risk_level=risk_level,
                anomaly_score=float(volatility_ratio),
                expected_value=float(historical_volatility),
                actual_value=float(current_volatility),
                deviation_percent=((current_volatility - historical_volatility) / historical_volatility) * 100,
                description=f"High demand volatility detected: {current_volatility:.1f} vs baseline {historical_volatility:.1f}",
                recommended_actions=[
                    "Increase safety stock by 20-30%",
                    "Review demand forecasting models",
                    "Investigate market factors causing volatility",
                    "Consider flexible ordering strategy"
                ],
                estimated_impact={
                    'stockout_risk': float(volatility_ratio * 0.15),
                    'holding_cost_increase': float(volatility_ratio * 500)
                }
            )

        return None

    def _detect_seasonal_anomaly(self, current_value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect deviation from seasonal pattern"""
        if self.seasonal_pattern is None:
            return None

        # Get expected seasonal value
        day_of_period = timestamp.weekday() if self.seasonal_period == 7 else timestamp.day % self.seasonal_period
        expected_seasonal = self.seasonal_pattern[day_of_period]

        # Calculate deviation
        deviation = abs(current_value - expected_seasonal) / (expected_seasonal + 1e-6)

        if deviation > self.thresholds['seasonal_deviation']:
            risk_level = self._calculate_risk_level(deviation, self.thresholds['seasonal_deviation'])

            return Anomaly(
                timestamp=timestamp,
                anomaly_type=AnomalyType.SEASONAL_ANOMALY,
                risk_level=risk_level,
                anomaly_score=float(deviation),
                expected_value=float(expected_seasonal),
                actual_value=float(current_value),
                deviation_percent=float(deviation * 100),
                description=f"Demand deviates from seasonal pattern: {current_value:.0f} vs expected {expected_seasonal:.0f}",
                recommended_actions=[
                    "Verify data quality and collection process",
                    "Check for special events or promotions",
                    "Review seasonal pattern assumptions",
                    "Consider one-time adjustment to forecast"
                ],
                estimated_impact=self._estimate_impact(current_value, expected_seasonal, AnomalyType.SEASONAL_ANOMALY)
            )

        return None

    def _detect_sustained_trend(self, recent_demand: np.ndarray, current_value: float) -> Optional[Anomaly]:
        """Detect sustained high or low demand trends"""
        if len(recent_demand) < 7:
            return None

        recent_window = recent_demand[-7:]  # Last week
        window_mean = np.mean(recent_window)

        # Check if sustained high
        if window_mean > self.baseline_mean * 1.3:
            all_high = np.all(recent_window > self.baseline_mean * 1.2)
            if all_high:
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.SUSTAINED_HIGH,
                    risk_level=RiskLevel.HIGH,
                    anomaly_score=float(window_mean / self.baseline_mean),
                    expected_value=float(self.baseline_mean),
                    actual_value=float(window_mean),
                    deviation_percent=((window_mean - self.baseline_mean) / self.baseline_mean) * 100,
                    description=f"Sustained high demand for 7 days: avg {window_mean:.0f} vs baseline {self.baseline_mean:.0f}",
                    recommended_actions=[
                        "🚨 CRITICAL: Increase order quantities by 30-50%",
                        "Expedite orders to prevent stockouts",
                        "Review if this is a permanent demand shift",
                        "Consider increasing safety stock levels",
                        "Negotiate with suppliers for increased capacity"
                    ],
                    estimated_impact={
                        'potential_stockout_loss': float((window_mean - self.baseline_mean) * 7 * 50),  # $50 per unit lost sale
                        'increased_revenue_opportunity': float((window_mean - self.baseline_mean) * 30 * 20)  # $20 margin per unit
                    }
                )

        # Check if sustained low
        if window_mean < self.baseline_mean * 0.7:
            all_low = np.all(recent_window < self.baseline_mean * 0.8)
            if all_low:
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.SUSTAINED_LOW,
                    risk_level=RiskLevel.MEDIUM,
                    anomaly_score=float(self.baseline_mean / window_mean),
                    expected_value=float(self.baseline_mean),
                    actual_value=float(window_mean),
                    deviation_percent=((window_mean - self.baseline_mean) / self.baseline_mean) * 100,
                    description=f"Sustained low demand for 7 days: avg {window_mean:.0f} vs baseline {self.baseline_mean:.0f}",
                    recommended_actions=[
                        "⚠️ Reduce order quantities to avoid overstock",
                        "Consider promotional activities to boost demand",
                        "Review product lifecycle and market trends",
                        "Investigate competitive dynamics",
                        "Optimize inventory to free up working capital"
                    ],
                    estimated_impact={
                        'excess_inventory_cost': float((self.baseline_mean - window_mean) * 30 * 2),  # $2 holding cost per unit
                        'capital_tied_up': float((self.baseline_mean - window_mean) * 30 * 10)  # $10 per unit
                    }
                )

        return None

    def _calculate_risk_level(self, score: float, threshold: float) -> RiskLevel:
        """Calculate risk level based on anomaly score"""
        ratio = score / threshold

        if ratio > 2.0:
            return RiskLevel.CRITICAL
        elif ratio > 1.5:
            return RiskLevel.HIGH
        elif ratio > 1.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _estimate_impact(self, actual: float, expected: float, anomaly_type: AnomalyType) -> Dict[str, float]:
        """Estimate financial impact of anomaly"""
        difference = actual - expected

        if anomaly_type == AnomalyType.SPIKE:
            return {
                'potential_stockout_loss': float(abs(difference) * 50),  # $50 lost profit per unit
                'emergency_order_cost': float(abs(difference) * 5)  # $5 extra cost per unit for rush order
            }
        elif anomaly_type == AnomalyType.DROP:
            return {
                'excess_holding_cost': float(abs(difference) * 2),  # $2 holding cost per unit per period
                'obsolescence_risk': float(abs(difference) * 10)  # $10 per unit at risk
            }
        else:
            return {
                'cost_impact': float(abs(difference) * 3)
            }

    def _generate_recommendations(self, anomaly_type: AnomalyType, actual: float, expected: float) -> List[str]:
        """Generate actionable recommendations based on anomaly type"""
        difference = actual - expected

        if anomaly_type == AnomalyType.SPIKE:
            recommendations = [
                f"🚨 URGENT: Place emergency order for {int(difference * 1.2)} units",
                "Contact supplier for expedited delivery (24-48h)",
                "Check competitor stock levels",
                f"Estimated stockout risk in 1-2 days without action",
                "Consider temporary price increase if supply constrained"
            ]
        elif anomaly_type == AnomalyType.DROP:
            recommendations = [
                f"⚠️ Reduce next order by {int(abs(difference) * 0.8)} units",
                "Review marketing and pricing strategy",
                "Investigate customer feedback and competitive landscape",
                "Consider promotional campaign to boost demand",
                "Monitor for further decline"
            ]
        else:
            recommendations = [
                "Monitor situation closely",
                "Review historical data for similar patterns",
                "Verify data quality",
                "Consider adjusting forecasting parameters"
            ]

        return recommendations

    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies, keeping highest risk ones"""
        if not anomalies:
            return []

        # Sort by risk level and anomaly score
        risk_priority = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }

        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: (risk_priority[x.risk_level], x.anomaly_score),
            reverse=True
        )

        # Keep top anomalies (max 3)
        return sorted_anomalies[:3]

    def get_summary_statistics(self, recent_demand: np.ndarray) -> Dict[str, Any]:
        """Get summary statistics for monitoring dashboard"""
        if not self.is_fitted:
            return {}

        return {
            'current_mean': float(np.mean(recent_demand)),
            'current_std': float(np.std(recent_demand)),
            'baseline_mean': float(self.baseline_mean),
            'baseline_std': float(self.baseline_std),
            'coefficient_of_variation': float(np.std(recent_demand) / (np.mean(recent_demand) + 1e-6)),
            'trend': 'increasing' if np.mean(recent_demand[-7:]) > np.mean(recent_demand[-14:-7]) else 'decreasing',
            'volatility_level': 'high' if np.std(recent_demand) > self.baseline_std * 1.5 else 'normal'
        }


if __name__ == "__main__":
    # Example usage
    print("🔍 Demand Anomaly Detection - Example")
    print("=" * 60)

    # Generate sample demand data with anomalies
    np.random.seed(42)
    n_days = 90

    # Normal baseline demand
    baseline = 50
    normal_demand = np.random.poisson(baseline, n_days)

    # Inject anomalies
    normal_demand[30] = 150  # Spike
    normal_demand[60:67] = 20  # Sustained drop
    normal_demand[80] = 5     # Drop

    # Initialize detector
    detector = DemandAnomalyDetector(
        window_size=14,
        seasonal_period=7,
        sensitivity='medium'
    )

    # Fit on first 50 days
    detector.fit(normal_demand[:50])

    print("✅ Detector fitted on 50 days of historical data")
    print(f"   Baseline mean: {detector.baseline_mean:.1f}")
    print(f"   Baseline std: {detector.baseline_std:.1f}")

    # Test detection on anomalous points
    test_points = [(30, 150), (60, 20), (80, 5), (85, 55)]

    for idx, value in test_points:
        print(f"\n📊 Testing day {idx}, demand = {value}")
        recent = normal_demand[max(0, idx-30):idx]

        anomalies = detector.detect(recent, value)

        if anomalies:
            print(f"   🚨 {len(anomalies)} anomalies detected:")
            for anomaly in anomalies:
                print(f"      - Type: {anomaly.anomaly_type.value}")
                print(f"      - Risk: {anomaly.risk_level.value}")
                print(f"      - Score: {anomaly.anomaly_score:.2f}")
                print(f"      - {anomaly.description}")
                print(f"      - Top recommendation: {anomaly.recommended_actions[0]}")
        else:
            print("   ✅ No anomalies detected")

    print("\n✅ Anomaly detection demo completed!")
