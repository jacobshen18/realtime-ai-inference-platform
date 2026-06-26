from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatencyMetrics:
    latencies_ms: list[float] = field(default_factory=list)

    def observe(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)

    @property
    def request_count(self) -> int:
        return len(self.latencies_ms)

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return round(sum(self.latencies_ms) / len(self.latencies_ms), 3)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
        return round(ordered[index], 3)
