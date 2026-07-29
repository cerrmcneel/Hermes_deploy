"""Unit tests for stuck loop detector circuit breaker."""

from hermes_deploy.stuck_loop import StuckLoopDetector


def test_stuck_loop_detection():
    """Verify circuit breaker triggers after N identical tool calls."""
    detector = StuckLoopDetector(threshold=3)

    # Turn 1
    assert not detector.record_turn("read_file", {"file_path": "a.txt"}, "content")
    # Turn 2
    assert not detector.record_turn("read_file", {"file_path": "a.txt"}, "content")
    # Turn 3 (Breach threshold)
    assert detector.record_turn("read_file", {"file_path": "a.txt"}, "content")


def test_no_stuck_loop_with_different_args():
    """Verify circuit breaker does not trigger when arguments change."""
    detector = StuckLoopDetector(threshold=3)

    assert not detector.record_turn("read_file", {"file_path": "a.txt"}, "content")
    assert not detector.record_turn("read_file", {"file_path": "b.txt"}, "content")
    assert not detector.record_turn("read_file", {"file_path": "c.txt"}, "content")
