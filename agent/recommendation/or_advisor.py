"""OR 策略建议器 — 将 MIP/LP/DP 输出转化为可执行建议"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from agent.decisions import OperationalDecision, StrategicDecision, TacticalDecision
from agent.coordinator import EnterpriseState


@dataclass
class ORRecommendation:
    layer: str
    period: str
    priority: str  # high | medium | low
    action: str
    rationale: str
    expected_impact: str


@dataclass
class ORAdviceReport:
    scenario_id: str
    policy_mode: str
    recommendations: List[ORRecommendation] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"=== OR 策略建议 [{self.scenario_id} / {self.policy_mode}] ==="]
        for r in self.recommendations:
            lines.append(f"[{r.priority.upper()}][{r.layer}·{r.period}] {r.action}")
            lines.append(f"  理由: {r.rationale}")
            lines.append(f"  预期: {r.expected_impact}")
        return "\n".join(lines)


class ORAdvisor:
    """基于运筹优化结果生成管理层可读建议。"""

    def advise_strategic(self, d: StrategicDecision, forecast_daily: float) -> List[ORRecommendation]:
        recs = []
        if not d.is_feasible:
            recs.append(ORRecommendation(
                "strategic", f"Q{d.quarter}", "high",
                "提高季度资本预算或外包产能",
                "MIP 在当前预算下无法满足产能目标",
                "避免持续缺货与订单流失",
            ))
            return recs

        cap_gap = forecast_daily - d.daily_capacity
        if cap_gap > 0:
            recs.append(ORRecommendation(
                "strategic", f"Q{d.quarter}", "high",
                f"扩产：当前产能 {d.daily_capacity:.0f}/日，缺口约 {cap_gap:.0f}/日",
                "预测日均需求高于可供应能力",
                "提升满足率、减少缺货罚金",
            ))
        elif d.open_factories:
            recs.append(ORRecommendation(
                "strategic", f"Q{d.quarter}", "medium",
                f"建厂 {d.open_factories}，产线 {d.lines_per_factory}，员工 {d.workforce}",
                f"MIP 最小成本方案，投资 ${d.investment_cost:,.0f}",
                "建立长期供应能力",
            ))
        else:
            recs.append(ORRecommendation(
                "strategic", f"Q{d.quarter}", "low",
                f"维持人力 {d.workforce} 人，暂不新建厂",
                "现有产能可覆盖预测，控制固定成本",
                "保持现金流稳定",
            ))
        return recs

    def advise_tactical(self, d: TacticalDecision, state: EnterpriseState) -> List[ORRecommendation]:
        if not d.is_feasible:
            return [ORRecommendation(
                "tactical", f"M{d.month}", "high",
                "放宽原料供应上限或下调产量目标",
                "LP 在 BOM/产能约束下不可行",
                "恢复生产连续性",
            )]
        prod = d.production_volume
        top = int(np.argmax(prod)) if len(prod) else 0
        return [ORRecommendation(
            "tactical", f"M{d.month}", "medium",
            f"产量分配 P{top+1}={prod[top]:.0f}，原料采购 {d.raw_material_procurement.sum():.0f} 单位",
            f"LP 边际利润最优，计划利润 ${d.profit:,.0f}",
            "按最优成本结构采购与排产",
        )]

    def advise_operational(
        self, d: OperationalDecision, state: EnterpriseState, demand: float
    ) -> List[ORRecommendation]:
        recs = []
        inv_days = state.inventory / max(demand, 1.0)
        if d.stockout > 0:
            recs.append(ORRecommendation(
                "operational", f"D{d.day}", "high",
                f"补货 {d.reorder_qty} 单位，优先履约 backlog",
                f"缺货 {d.stockout:.0f}，库存仅够 {inv_days:.1f} 天",
                "降低缺货损失",
            ))
        if state.equipment_health < 0.85 and not d.maintain_equipment:
            recs.append(ORRecommendation(
                "operational", f"D{d.day}", "medium",
                "安排设备预防性维护",
                f"设备健康度 {state.equipment_health:.0%}，故障风险上升",
                "避免产能骤降",
            ))
        if not recs:
            recs.append(ORRecommendation(
                "operational", f"D{d.day}", "low",
                f"维持补货 {d.reorder_qty}，库存 {d.inventory_end:.0f}",
                "DP 当前状态均衡",
                "稳定运营",
            ))
        return recs

    def build_report(
        self,
        scenario_id: str,
        policy_mode: str,
        strategic: List[StrategicDecision],
        tactical: List[TacticalDecision],
        sample_ops: Optional[List[OperationalDecision]] = None,
    ) -> ORAdviceReport:
        recs: List[ORRecommendation] = []
        for s in strategic:
            forecast = s.daily_capacity * 1.25 if s.is_feasible else s.daily_capacity * 2
            recs.extend(self.advise_strategic(s, forecast))
        for t in tactical:
            recs.extend(self.advise_tactical(t, EnterpriseState()))
        if sample_ops:
            st = EnterpriseState()
            for op in sample_ops:
                recs.extend(self.advise_operational(op, st, 250.0))
        return ORAdviceReport(scenario_id, policy_mode, recs)
