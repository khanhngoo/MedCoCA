from coca_med.eval.metrics import aggregate_metrics, brier_score, expected_calibration_error


def test_brier_score() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_expected_calibration_error_perfect() -> None:
    assert expected_calibration_error([1.0, 0.0], [1, 0], bins=2) == 0.0


def test_aggregate_metrics() -> None:
    metrics = aggregate_metrics(
        confidences=[0.9, 0.2, None],
        correctness=[1, 0, 1],
        token_to_confidence=[8, 9, None],
        bins=5,
    )

    assert metrics.count == 3
    assert metrics.accuracy == 2 / 3
    assert metrics.confidence_success_rate == 2 / 3
    assert metrics.token_to_confidence == 8.5
