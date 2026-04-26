from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS = ["accuracy", "fp", "fn", "fps", "run_time", "precision", "recall", "f1_score"]

