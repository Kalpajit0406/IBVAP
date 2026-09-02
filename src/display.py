from __future__ import annotations

import cv2
import numpy as np

from .detector import StreamResult
from .risk_engine import RiskAssessment

_COLOURS = {
    "Normal":   (0, 200, 0),
    "High":     (0, 165, 255),
    "Critical": (0, 0, 255),
}
_HEADER_H = 48


class GridDisplay:
    """
    Arranges all active stream frames in a responsive grid.
    Each cell shows live detections + a risk-score header bar.
    """

    def __init__(self, config: dict) -> None:
        d = config["display"]
        self._cols = d.get("grid_cols", 3)
        self._cell_w = d.get("cell_width", 640)
        self._cell_h = d.get("cell_height", 360)
        self._window = "IBVAP — Multi-Stream Monitor"
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, self._cell_w * min(self._cols, 3), self._cell_h * 2)

    def show(
        self,
        results: list[StreamResult],
        risks: dict[int, RiskAssessment],
    ) -> bool:
        """Render the grid. Returns False when the user presses 'q'."""
        cells = []
        for sr in results:
            base = sr.annotated_frame if sr.annotated_frame is not None else sr.frame
            cell = self._build_cell(base, sr, risks.get(sr.cam_id))
            cells.append(cell)

        if not cells:
            grid = np.zeros((self._cell_h, self._cell_w, 3), dtype=np.uint8)
        else:
            grid = self._make_grid(cells)

        cv2.imshow(self._window, grid)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_cell(
        self,
        frame: np.ndarray,
        sr: StreamResult,
        risk: RiskAssessment | None,
    ) -> np.ndarray:
        # Resize to uniform cell size
        cell = cv2.resize(frame, (self._cell_w, self._cell_h - _HEADER_H))

        # Header bar
        header = np.zeros((_HEADER_H, self._cell_w, 3), dtype=np.uint8)
        header[:] = (20, 20, 20)

        level = risk.level if risk else "Normal"
        colour = _COLOURS[level]

        # Left: camera name
        cv2.putText(header, f"CAM-{sr.cam_id:02d}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # Centre: risk score
        if risk:
            score_txt = f"Risk {risk.score:5.1f}  [{level}]"
            cv2.putText(header, score_txt, (140, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)

        # Right: person / vehicle counts
        counts_txt = f"P:{sr.person_count}  V:{sr.vehicle_count}"
        tw, _ = cv2.getTextSize(counts_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0], None
        cv2.putText(header, counts_txt, (self._cell_w - tw[0] - 10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

        cell_full = np.vstack([header, cell])

        # Red border on Critical
        if level == "Critical":
            cv2.rectangle(cell_full, (0, 0),
                          (self._cell_w - 1, self._cell_h - 1), colour, 4)

        return cell_full

    def _make_grid(self, cells: list[np.ndarray]) -> np.ndarray:
        cols = min(self._cols, len(cells))
        rows = (len(cells) + cols - 1) // cols

        # Pad to full grid
        blank = np.full((self._cell_h, self._cell_w, 3), 30, dtype=np.uint8)
        while len(cells) < rows * cols:
            cells.append(blank)

        rows_img = [
            np.hstack(cells[r * cols: (r + 1) * cols])
            for r in range(rows)
        ]
        return np.vstack(rows_img)

    def close(self) -> None:
        cv2.destroyAllWindows()
