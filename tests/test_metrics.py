from realtime_ai_platform.metrics import LatencyMetrics


def test_latency_metrics_empty_state() -> None:
    metrics = LatencyMetrics()

    assert metrics.request_count == 0
    assert metrics.average_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0


def test_latency_metrics_observations() -> None:
    metrics = LatencyMetrics()
    metrics.observe(10)
    metrics.observe(20)
    metrics.observe(30)

    assert metrics.request_count == 3
    assert metrics.average_latency_ms == 20
    assert metrics.p95_latency_ms == 20
