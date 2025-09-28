from __future__ import annotations

import time
from typing import Optional


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"


class Progress:
    def __init__(self, total: int, name: str = "progress", every_s: float = 1.0):
        self.total = max(1, int(total))
        self.name = name
        self.every = float(every_s)
        self.start = time.time()
        self.last = self.start
        self.last_pct = -1.0

    def update(self, n: int, info: Optional[str] = None):
        now = time.time()
        if now - self.last < self.every and (100.0 * n / self.total) - self.last_pct < 1.0:
            return
        self.last = now
        pct = 100.0 * n / self.total
        self.last_pct = pct
        elapsed = now - self.start
        rate = n / max(elapsed, 1e-6)
        remain = max(0, self.total - n)
        eta = remain / max(rate, 1e-6)
        extra = f" {info}" if info else ""
        print(f"[{pct:5.1f}%] {self.name} {n}/{self.total} elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)} rate={rate:.1f}/s{extra}")

    def finalize(self):
        total = time.time() - self.start
        print(f"[100.0%] {self.name} done in {fmt_duration(total)}")

