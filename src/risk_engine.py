from __future__ import annotations

import datetime
from dataclasses import dataclass

from .detector import StreamResult


@dataclass
class RiskAssessment:
    cam_id: int
    score: float        # 0–100
    level: str          # "Normal" | "High" | "Critical"
    zone: float
    time_of_day: float
    behaviour: float
    weapon: bool = False   # a weapon-ready posture was detected this frame


class RiskEngine:
    """
    Weighted risk scoring:  score = (zone×0.4 + time×0.2 + behaviour×0.4) × 100
    Each component returns a value in [0, 1].
    """

    def __init__(self, config: dict) -> None:
        r = config["risk"]
        self._crit = r["threshold_critical"]
        self._high = r["threshold_high"]
        w = r["weights"]
        self._w_zone = w["zone"]
        self._w_time = w["time_of_day"]
        self._w_beh = w["behaviour"]
        self._stream_cfg = {s["id"]: s for s in config["streams"]}

    def assess(self, sr: StreamResult) -> RiskAssessment:
        zone = self._zone(sr)
        tod = self._time_of_day()
        beh = self._behaviour(sr)

        raw = zone * self._w_zone + tod * self._w_time + beh * self._w_beh
        score = min(100.0, max(0.0, raw * 100))

        if score >= self._crit:
            level = "Critical"
        elif score >= self._high:
            level = "High"
        else:
            level = "Normal"

        # A confirmed weapon-ready posture is the highest threat there is — it
        # must never be gated by the weighted formula (which only reaches
        # Critical at night). Force it.
        weapon = any(
            d.is_person and d.posture is not None and d.posture.chest_aim
            for d in sr.detections
        )
        if weapon:
            level = "Critical"
            score = max(score, 92.0)

        return RiskAssessment(sr.cam_id, score, level, zone, tod, beh, weapon)

    # ── Component scorers ─────────────────────────────────────────────────────

    def _zone(self, sr: StreamResult) -> float:
        sensitivity = self._stream_cfg.get(sr.cam_id, {}).get("zone_sensitivity", 0.5)
        return sensitivity if (sr.person_count > 0 or sr.vehicle_count > 0) else 0.0

    def _time_of_day(self) -> float:
        hour = datetime.datetime.now().hour
        if 22 <= hour or hour < 5:
            return 1.0    # night
        if 5 <= hour < 7 or 20 <= hour < 22:
            return 0.6    # twilight
        return 0.2        # daytime

    def _behaviour(self, sr: StreamResult) -> float:
        """
        Combines person/vehicle count with per-person posture anomalies.
        PostureFlags are attached to Detection.posture by the detector.
        Falls back gracefully to count-only when pose is disabled.
        """
        count = sr.person_count + sr.vehicle_count
        if count == 0:
            return 0.0

        # Base score from head count
        if count == 1:
            base = 0.3
        elif count <= 3:
            base = 0.6
        else:
            base = 0.8

        # Posture boost: highest anomaly weight among all persons this frame
        posture_boost = max(
            (d.posture.risk_weight
             for d in sr.detections
             if d.is_person and d.posture is not None),
            default=0.0,
        )

        # Posture can only push the score up, never down
        return min(1.0, base + posture_boost * 0.5)
