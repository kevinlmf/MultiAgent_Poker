"""ML/DL 策略建议 — 解释模型预测"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agent.decisions import StrategicDecision, TacticalDecision
from agent.recommendation.or_advisor import ORRecommendation


@dataclass
class MLAdviceReport:
    scenario_id: str
    policy_mode: str
    recommendations: List[ORRecommendation] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"=== ML/DL 策略建议 [{self.scenario_id}] ==="]
        for r in self.recommendations:
            lines.append(f"[{r.priority.upper()}][{r.layer}·{r.period}] {r.action}")
            lines.append(f"  模型依据: {r.rationale}")
        return "\n".join(lines)


class MLAdvisor:
    def build_report(
        self, scenario_id: str, strategic: List[StrategicDecision], tactical: List[TacticalDecision]
    ) -> MLAdviceReport:
        recs = []
        for s in strategic:
            recs.append(ORRecommendation(
                "strategic", f"Q{s.quarter}", "medium",
                f"ML 预测日产能 {s.daily_capacity:.0f}，员工 {s.workforce}",
                "RandomForest 基于历史需求/波动特征",
                f"投资约 ${s.investment_cost:,.0f}",
            ))
        for t in tactical[:4]:
            recs.append(ORRecommendation(
                "tactical", f"M{t.month}", "medium",
                f"GBR 预测产量 {t.production_volume.round(0).tolist()}",
                "梯度提升树 + 指数平滑融合",
                f"计划利润 ${t.profit:,.0f}",
            ))
        recs.append(ORRecommendation(
            "operational", "daily", "medium",
            "MLP(64,32) 预测每日补货量与维护开关",
            "深度学习 MLP 监督学习",
            "降低缺货并自适应设备健康",
        ))
        return MLAdviceReport(scenario_id, "ml", recs)
