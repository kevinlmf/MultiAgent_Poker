"""
Intelligent Contingency Planning System

Generates multiple actionable contingency plans for different risk scenarios
with cost-benefit analysis and automated execution recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .anomaly_detector import Anomaly, AnomalyType, RiskLevel


class ActionType(Enum):
    """Types of contingency actions"""
    EMERGENCY_ORDER = "emergency_order"
    EXPEDITE_EXISTING = "expedite_existing_order"
    REDUCE_ORDER = "reduce_order_quantity"
    CANCEL_ORDER = "cancel_pending_order"
    ADJUST_SAFETY_STOCK = "adjust_safety_stock"
    PROMOTIONAL_CAMPAIGN = "run_promotion"
    PRICE_ADJUSTMENT = "adjust_pricing"
    ALTERNATIVE_SUPPLIER = "use_alternative_supplier"
    CUSTOMER_RATIONING = "implement_rationing"
    DO_NOTHING = "monitor_and_wait"


@dataclass
class ContingencyAction:
    """Represents a specific action in a contingency plan"""
    action_type: ActionType
    description: str
    parameters: Dict[str, Any]  # e.g., {'quantity': 100, 'delivery_days': 2}
    estimated_cost: float
    implementation_time: int  # hours
    effectiveness_score: float  # 0-1
    dependencies: List[str]  # Other actions this depends on


@dataclass
class ContingencyPlan:
    """A complete contingency plan with multiple actions"""
    plan_id: str
    plan_name: str
    risk_scenario: str
    actions: List[ContingencyAction]
    total_cost: float
    total_benefit: float
    net_benefit: float  # benefit - cost
    roi: float  # return on investment
    implementation_time: int  # hours
    success_probability: float
    risk_mitigation_level: float  # 0-1, how much risk is reduced
    recommendation_priority: int  # 1 (best) to 3 (worst)
    detailed_explanation: str


class ContingencyPlanner:
    """
    Intelligent system that generates optimal contingency plans
    based on detected risks and anomalies
    """

    def __init__(self,
                 holding_cost: float = 2.0,
                 stockout_cost: float = 50.0,
                 ordering_cost: float = 50.0,
                 unit_cost: float = 10.0,
                 unit_price: float = 20.0,
                 lead_time_normal: int = 7,
                 lead_time_expedited: int = 2):
        """
        Initialize contingency planner

        Args:
            holding_cost: Cost per unit per day
            stockout_cost: Lost profit per stockout unit
            ordering_cost: Fixed cost per order
            unit_cost: Purchase cost per unit
            unit_price: Selling price per unit
            lead_time_normal: Normal delivery lead time (days)
            lead_time_expedited: Expedited delivery lead time (days)
        """
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self.unit_price = unit_price
        self.unit_margin = unit_price - unit_cost
        self.lead_time_normal = lead_time_normal
        self.lead_time_expedited = lead_time_expedited

    def generate_plans(self,
                      anomaly: Anomaly,
                      current_inventory: float,
                      pending_orders: float,
                      demand_forecast: np.ndarray) -> List[ContingencyPlan]:
        """
        Generate multiple contingency plans for a detected anomaly

        Args:
            anomaly: Detected anomaly from AnomalyDetector
            current_inventory: Current inventory level
            pending_orders: Orders in transit
            demand_forecast: Forecasted demand for next periods

        Returns:
            List of contingency plans, sorted by ROI
        """
        plans = []

        if anomaly.anomaly_type == AnomalyType.SPIKE:
            plans = self._generate_spike_plans(anomaly, current_inventory, pending_orders, demand_forecast)
        elif anomaly.anomaly_type == AnomalyType.DROP:
            plans = self._generate_drop_plans(anomaly, current_inventory, pending_orders, demand_forecast)
        elif anomaly.anomaly_type == AnomalyType.SUSTAINED_HIGH:
            plans = self._generate_sustained_high_plans(anomaly, current_inventory, pending_orders, demand_forecast)
        elif anomaly.anomaly_type == AnomalyType.SUSTAINED_LOW:
            plans = self._generate_sustained_low_plans(anomaly, current_inventory, pending_orders, demand_forecast)
        elif anomaly.anomaly_type == AnomalyType.VOLATILITY:
            plans = self._generate_volatility_plans(anomaly, current_inventory, pending_orders, demand_forecast)

        # Sort by net benefit (ROI)
        plans.sort(key=lambda p: p.net_benefit, reverse=True)

        # Assign priority
        for i, plan in enumerate(plans):
            plan.recommendation_priority = i + 1

        return plans[:3]  # Return top 3 plans

    def _generate_spike_plans(self,
                             anomaly: Anomaly,
                             current_inventory: float,
                             pending_orders: float,
                             demand_forecast: np.ndarray) -> List[ContingencyPlan]:
        """Generate plans for demand spike scenario"""
        plans = []

        shortage = max(0, anomaly.actual_value - current_inventory - pending_orders)
        forecast_next_week = np.sum(demand_forecast[:7])

        # Plan A: Emergency expedited order (aggressive)
        emergency_qty = int(shortage + forecast_next_week * 0.3)  # Cover shortage + buffer
        expedite_cost = emergency_qty * (self.unit_cost * 1.5) + self.ordering_cost * 2  # 50% premium + rush fee
        stockout_prevented = emergency_qty
        benefit = stockout_prevented * self.stockout_cost

        plan_a = ContingencyPlan(
            plan_id="PLAN_A_EMERGENCY",
            plan_name="🚨 Emergency Expedited Order",
            risk_scenario=f"Demand spike: {anomaly.actual_value:.0f} units (vs expected {anomaly.expected_value:.0f})",
            actions=[
                ContingencyAction(
                    action_type=ActionType.EMERGENCY_ORDER,
                    description=f"Place urgent order for {emergency_qty} units with 48h delivery",
                    parameters={
                        'quantity': emergency_qty,
                        'delivery_days': self.lead_time_expedited,
                        'premium_rate': 1.5
                    },
                    estimated_cost=expedite_cost,
                    implementation_time=1,  # 1 hour to place order
                    effectiveness_score=0.95,
                    dependencies=[]
                )
            ],
            total_cost=expedite_cost,
            total_benefit=benefit,
            net_benefit=benefit - expedite_cost,
            roi=(benefit - expedite_cost) / expedite_cost if expedite_cost > 0 else 0,
            implementation_time=1,
            success_probability=0.90,
            risk_mitigation_level=0.95,
            recommendation_priority=1,
            detailed_explanation=f"""
**Plan A: Emergency Expedited Order**

✅ **Benefits:**
- Prevents {int(stockout_prevented)} units of potential stockouts
- Maintains customer satisfaction
- Captures revenue opportunity: ${benefit:,.0f}
- Delivery in {self.lead_time_expedited} days

⚠️ **Costs:**
- Order cost: ${expedite_cost:,.0f}
- 50% premium on unit costs for expedited delivery
- Rush ordering fee

💡 **Recommendation:**
{"⭐ HIGHLY RECOMMENDED - Best for critical situations where customer retention is priority" if anomaly.risk_level == RiskLevel.CRITICAL else "Consider if stockout risk is high"}

📊 **Financial Impact:**
- ROI: {(benefit - expedite_cost) / expedite_cost * 100:.1f}%
- Break-even: {int(expedite_cost / self.stockout_cost)} units prevented stockout
- Implementation time: 1 hour + {self.lead_time_expedited} days delivery
            """
        )
        plans.append(plan_a)

        # Plan B: Moderate order + customer management (balanced)
        moderate_qty = int(shortage * 0.7)
        moderate_cost = moderate_qty * self.unit_cost * 1.2 + self.ordering_cost  # 20% premium
        remaining_stockout = shortage - moderate_qty
        moderate_benefit = moderate_qty * self.stockout_cost - remaining_stockout * (self.stockout_cost * 0.5)  # 50% loss from rationing

        plan_b = ContingencyPlan(
            plan_id="PLAN_B_BALANCED",
            plan_name="⚖️ Balanced Order + Customer Rationing",
            risk_scenario=f"Demand spike: {anomaly.actual_value:.0f} units",
            actions=[
                ContingencyAction(
                    action_type=ActionType.EMERGENCY_ORDER,
                    description=f"Order {moderate_qty} units with 3-4 day delivery",
                    parameters={'quantity': moderate_qty, 'delivery_days': 3, 'premium_rate': 1.2},
                    estimated_cost=moderate_cost,
                    implementation_time=2,
                    effectiveness_score=0.70,
                    dependencies=[]
                ),
                ContingencyAction(
                    action_type=ActionType.CUSTOMER_RATIONING,
                    description="Implement purchase limits: 2 units per customer",
                    parameters={'limit_per_customer': 2, 'duration_days': 5},
                    estimated_cost=0,
                    implementation_time=1,
                    effectiveness_score=0.60,
                    dependencies=[]
                )
            ],
            total_cost=moderate_cost,
            total_benefit=moderate_benefit,
            net_benefit=moderate_benefit - moderate_cost,
            roi=(moderate_benefit - moderate_cost) / moderate_cost if moderate_cost > 0 else 0,
            implementation_time=2,
            success_probability=0.75,
            risk_mitigation_level=0.70,
            recommendation_priority=2,
            detailed_explanation=f"""
**Plan B: Balanced Approach**

✅ **Benefits:**
- Lower cost than full expedited order
- Manages customer expectations
- Estimated savings: ${moderate_cost - expedite_cost:,.0f} vs Plan A

⚠️ **Trade-offs:**
- Some customers may be dissatisfied with limits
- Partial stockout risk: ~{int(remaining_stockout)} units
- Longer delivery time (3-4 days)

💡 **Best for:**
- Budget-conscious scenarios
- When customer base tolerates rationing
- Non-critical seasonal spikes

📊 **Financial Impact:**
- ROI: {(moderate_benefit - moderate_cost) / moderate_cost * 100:.1f}%
- Cost savings vs Plan A: ${expedite_cost - moderate_cost:,.0f}
            """
        )
        plans.append(plan_b)

        # Plan C: Monitor + small adjustment (conservative)
        small_qty = int(shortage * 0.3)
        small_cost = small_qty * self.unit_cost + self.ordering_cost
        small_benefit = small_qty * self.stockout_cost * 0.5

        plan_c = ContingencyPlan(
            plan_id="PLAN_C_CONSERVATIVE",
            plan_name="📊 Monitor + Minimal Adjustment",
            risk_scenario=f"Demand spike: {anomaly.actual_value:.0f} units",
            actions=[
                ContingencyAction(
                    action_type=ActionType.EMERGENCY_ORDER,
                    description=f"Small safety order: {small_qty} units",
                    parameters={'quantity': small_qty, 'delivery_days': self.lead_time_normal},
                    estimated_cost=small_cost,
                    implementation_time=2,
                    effectiveness_score=0.30,
                    dependencies=[]
                ),
                ContingencyAction(
                    action_type=ActionType.DO_NOTHING,
                    description="Monitor demand for 48h before further action",
                    parameters={'monitoring_period_hours': 48},
                    estimated_cost=0,
                    implementation_time=0,
                    effectiveness_score=0.20,
                    dependencies=[]
                )
            ],
            total_cost=small_cost,
            total_benefit=small_benefit,
            net_benefit=small_benefit - small_cost,
            roi=(small_benefit - small_cost) / small_cost if small_cost > 0 else 0,
            implementation_time=2,
            success_probability=0.40,
            risk_mitigation_level=0.30,
            recommendation_priority=3,
            detailed_explanation=f"""
**Plan C: Conservative Wait-and-See**

✅ **Benefits:**
- Minimal upfront cost
- Avoids overreaction to potentially temporary spike
- Normal delivery timeline

⚠️ **Risks:**
- High stockout probability: ~{int(shortage - small_qty)} units
- May lose sales to competitors
- Customer dissatisfaction risk

💡 **Only consider if:**
- Spike is suspected to be one-time event
- High confidence demand will normalize
- Low customer retention impact

⚠️ **WARNING:** This plan has 30% risk mitigation - consider higher priority plans
            """
        )
        plans.append(plan_c)

        return plans

    def _generate_drop_plans(self,
                            anomaly: Anomaly,
                            current_inventory: float,
                            pending_orders: float,
                            demand_forecast: np.ndarray) -> List[ContingencyPlan]:
        """Generate plans for demand drop scenario"""
        plans = []

        excess = current_inventory + pending_orders - anomaly.actual_value * 7  # 7 days of low demand
        forecast_next_month = np.sum(demand_forecast[:30])

        # Plan A: Promotional campaign (proactive)
        promo_cost = 500 + excess * 0.5  # Marketing cost + discount
        promo_benefit = excess * self.unit_margin * 0.7  # Sell 70% of excess at discounted price

        plan_a = ContingencyPlan(
            plan_id="PLAN_A_PROMOTION",
            plan_name="🎯 Aggressive Promotional Campaign",
            risk_scenario=f"Demand drop: {anomaly.actual_value:.0f} units (vs expected {anomaly.expected_value:.0f})",
            actions=[
                ContingencyAction(
                    action_type=ActionType.PROMOTIONAL_CAMPAIGN,
                    description="Run 15% off promotion + social media campaign",
                    parameters={'discount_rate': 0.15, 'marketing_budget': 500, 'duration_days': 14},
                    estimated_cost=promo_cost,
                    implementation_time=4,
                    effectiveness_score=0.70,
                    dependencies=[]
                ),
                ContingencyAction(
                    action_type=ActionType.REDUCE_ORDER,
                    description=f"Reduce pending order by {int(pending_orders * 0.3)} units",
                    parameters={'reduction_quantity': int(pending_orders * 0.3)},
                    estimated_cost=50,  # Cancellation fee
                    implementation_time=1,
                    effectiveness_score=0.80,
                    dependencies=[]
                )
            ],
            total_cost=promo_cost + 50,
            total_benefit=promo_benefit,
            net_benefit=promo_benefit - (promo_cost + 50),
            roi=(promo_benefit - (promo_cost + 50)) / (promo_cost + 50) if promo_cost > 0 else 0,
            implementation_time=4,
            success_probability=0.65,
            risk_mitigation_level=0.70,
            recommendation_priority=1,
            detailed_explanation=f"""
**Plan A: Promotional Campaign**

✅ **Benefits:**
- Proactively addresses excess inventory
- Generates revenue: ${promo_benefit:,.0f}
- Maintains brand visibility
- Attracts new customers

⚠️ **Costs:**
- Promotion cost: ${promo_cost:,.0f}
- Margin reduction from discount

💡 **Best for:**
- Seasonal or perishable goods
- When market share growth is valuable
- Excess inventory > 30 days

📊 **Expected Outcomes:**
- Move {int(excess * 0.7)} units in 14 days
- ROI: {(promo_benefit - (promo_cost + 50)) / (promo_cost + 50) * 100:.1f}%
- Free up ${excess * 0.7 * self.unit_cost:,.0f} in working capital
            """
        )
        plans.append(plan_a)

        # Plan B: Reduce orders + hold inventory (moderate)
        holding_duration = 30  # days
        holding_cost_total = excess * self.holding_cost * holding_duration

        plan_b = ContingencyPlan(
            plan_id="PLAN_B_REDUCE_HOLD",
            plan_name="📦 Reduce Orders + Hold Inventory",
            risk_scenario=f"Demand drop: {anomaly.actual_value:.0f} units",
            actions=[
                ContingencyAction(
                    action_type=ActionType.REDUCE_ORDER,
                    description=f"Cancel/reduce {int(pending_orders * 0.5)} units from pending orders",
                    parameters={'reduction_quantity': int(pending_orders * 0.5)},
                    estimated_cost=100,  # Cancellation fee
                    implementation_time=1,
                    effectiveness_score=0.80,
                    dependencies=[]
                ),
                ContingencyAction(
                    action_type=ActionType.ADJUST_SAFETY_STOCK,
                    description="Reduce safety stock levels by 30%",
                    parameters={'reduction_percent': 0.30},
                    estimated_cost=0,
                    implementation_time=1,
                    effectiveness_score=0.60,
                    dependencies=[]
                )
            ],
            total_cost=holding_cost_total + 100,
            total_benefit=0,  # No immediate benefit, just cost avoidance
            net_benefit=-holding_cost_total - 100,
            roi=-1.0,  # Negative ROI
            implementation_time=1,
            success_probability=0.80,
            risk_mitigation_level=0.60,
            recommendation_priority=2,
            detailed_explanation=f"""
**Plan B: Conservative Inventory Management**

✅ **Benefits:**
- No discount on prices (maintain margins)
- Suitable if demand expected to recover
- Lower risk than promotions

⚠️ **Costs:**
- Holding cost: ${holding_cost_total:,.0f} for {holding_duration} days
- Capital tied up: ${excess * self.unit_cost:,.0f}
- Order cancellation: $100

💡 **Best for:**
- Temporary demand dips
- High-margin products
- When promotional activity would damage brand

⚠️ **Note:** This plan has negative immediate ROI but may be worth it if demand recovers
            """
        )
        plans.append(plan_b)

        # Plan C: Do nothing + monitor
        plan_c = ContingencyPlan(
            plan_id="PLAN_C_MONITOR",
            plan_name="👀 Monitor and Wait",
            risk_scenario=f"Demand drop: {anomaly.actual_value:.0f} units",
            actions=[
                ContingencyAction(
                    action_type=ActionType.DO_NOTHING,
                    description="Monitor demand trends for 7 days before action",
                    parameters={'monitoring_period_days': 7},
                    estimated_cost=0,
                    implementation_time=0,
                    effectiveness_score=0.20,
                    dependencies=[]
                )
            ],
            total_cost=excess * self.holding_cost * 7,  # 1 week holding cost
            total_benefit=0,
            net_benefit=-excess * self.holding_cost * 7,
            roi=-1.0,
            implementation_time=0,
            success_probability=0.30,
            risk_mitigation_level=0.20,
            recommendation_priority=3,
            detailed_explanation=f"""
**Plan C: Wait-and-See Approach**

✅ **Benefits:**
- No immediate action needed
- May resolve naturally
- No promotional discount

⚠️ **Risks:**
- Accumulating holding costs: ${excess * self.holding_cost * 30:,.0f}/month
- Risk of obsolescence
- Capital remains tied up

⚠️ **WARNING:** Only suitable if very confident demand will recover quickly
            """
        )
        plans.append(plan_c)

        return plans

    def _generate_sustained_high_plans(self, anomaly, current_inventory, pending_orders, demand_forecast):
        """Generate plans for sustained high demand"""
        # Similar structure to spike plans but with longer-term adjustments
        plans = []
        # Implementation similar to spike plans
        return plans

    def _generate_sustained_low_plans(self, anomaly, current_inventory, pending_orders, demand_forecast):
        """Generate plans for sustained low demand"""
        # Similar structure to drop plans but with longer-term adjustments
        plans = []
        # Implementation similar to drop plans
        return plans

    def _generate_volatility_plans(self, anomaly, current_inventory, pending_orders, demand_forecast):
        """Generate plans for high volatility scenario"""
        plans = []
        # Focus on increasing buffer stocks and flexible ordering
        return plans

    def format_plan_for_display(self, plan: ContingencyPlan) -> str:
        """Format contingency plan for user-friendly display"""
        output = f"""
{'='*70}
{plan.plan_name}
{'='*70}

📋 SCENARIO: {plan.risk_scenario}

💰 FINANCIAL SUMMARY:
   Total Cost:        ${plan.total_cost:,.0f}
   Expected Benefit:  ${plan.total_benefit:,.0f}
   Net Benefit:       ${plan.net_benefit:,.0f}
   ROI:               {plan.roi * 100:.1f}%

⏱️  IMPLEMENTATION:
   Time Required:     {plan.implementation_time} hours
   Success Probability: {plan.success_probability * 100:.0f}%
   Risk Mitigation:   {plan.risk_mitigation_level * 100:.0f}%

📝 ACTIONS TO TAKE:
"""
        for i, action in enumerate(plan.actions, 1):
            output += f"\n   {i}. {action.description}"
            output += f"\n      Cost: ${action.estimated_cost:,.0f} | Effectiveness: {action.effectiveness_score * 100:.0f}%"

        output += f"\n\n{plan.detailed_explanation}"
        output += "\n" + "="*70 + "\n"

        return output


if __name__ == "__main__":
    # Example usage
    print("🎯 Contingency Planning System - Example")
    print("=" * 70)

    from .anomaly_detector import Anomaly, AnomalyType, RiskLevel

    # Create example anomaly
    spike_anomaly = Anomaly(
        timestamp=datetime.now(),
        anomaly_type=AnomalyType.SPIKE,
        risk_level=RiskLevel.HIGH,
        anomaly_score=3.5,
        expected_value=50.0,
        actual_value=150.0,
        deviation_percent=200.0,
        description="Demand spike detected",
        recommended_actions=[],
        estimated_impact={'potential_stockout_loss': 5000}
    )

    # Initialize planner
    planner = ContingencyPlanner(
        holding_cost=2.0,
        stockout_cost=50.0,
        unit_cost=10.0,
        unit_price=20.0
    )

    # Generate plans
    current_inventory = 30
    pending_orders = 20
    forecast = np.array([60, 55, 58, 62, 65, 70, 68])

    print(f"\n📊 Current Situation:")
    print(f"   Inventory: {current_inventory} units")
    print(f"   Pending Orders: {pending_orders} units")
    print(f"   Anomaly: {spike_anomaly.description}")

    plans = planner.generate_plans(spike_anomaly, current_inventory, pending_orders, forecast)

    print(f"\n✅ Generated {len(plans)} contingency plans:\n")

    for plan in plans:
        print(planner.format_plan_for_display(plan))

    print("\n✅ Contingency planning demo completed!")
