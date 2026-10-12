"""Unit tests for motion clustering, anomaly detection, and MotionField."""

from __future__ import annotations

import math

import pytest

from opticflow_cv.core.models import MotionVector
from opticflow_cv.motion.anomaly import MotionAnomalyDetector
from opticflow_cv.motion.clustering import MotionClusterer
from opticflow_cv.motion.motion_field import MotionField


class TestMotionClusterer:
    """Tests for spatial-velocity clustering of motion vectors."""

    def test_single_coherent_cluster(self):
        # 5 vectors clustered closely around (50, 50) moving right (vx=4, vy=0)
        vecs = [
            MotionVector(x=48.0, y=49.0, vx=4.1, vy=0.1),
            MotionVector(x=50.0, y=50.0, vx=4.0, vy=0.0),
            MotionVector(x=52.0, y=51.0, vx=3.9, vy=-0.1),
            MotionVector(x=49.0, y=52.0, vx=4.0, vy=0.0),
        ]
        clusterer = MotionClusterer(eps=15.0, min_samples=3)
        clusters = clusterer.cluster(vecs)

        assert len(clusters) == 1
        c = clusters[0]
        assert c.member_count == 4
        assert c.centroid[0] == pytest.approx(49.75, abs=1.0)
        assert c.centroid[1] == pytest.approx(50.5, abs=1.0)
        assert c.average_velocity[0] == pytest.approx(4.0, abs=0.2)
        assert c.average_speed == pytest.approx(4.0, abs=0.2)
        assert c.bounding_box[0] == 48
        assert c.bounding_box[2] == 52

    def test_two_distinct_clusters(self):
        # Object 1 near (20, 20) moving right (+vx)
        obj1 = [
            MotionVector(x=19.0, y=20.0, vx=5.0, vy=0.0),
            MotionVector(x=20.0, y=21.0, vx=5.1, vy=0.1),
            MotionVector(x=21.0, y=19.0, vx=4.9, vy=-0.1),
        ]
        # Object 2 far away near (100, 100) moving down (+vy)
        obj2 = [
            MotionVector(x=99.0, y=100.0, vx=0.0, vy=6.0),
            MotionVector(x=100.0, y=101.0, vx=0.1, vy=5.9),
            MotionVector(x=101.0, y=99.0, vx=-0.1, vy=6.1),
        ]
        clusterer = MotionClusterer(eps=15.0, min_samples=3)
        clusters = clusterer.cluster(obj1 + obj2)

        assert len(clusters) == 2

    def test_empty_vectors_return_empty(self):
        clusterer = MotionClusterer()
        assert clusterer.cluster([]) == []

    def test_divergent_velocities_not_clustered(self):
        # Spatially close vectors moving in opposite directions
        vecs = [
            MotionVector(x=30.0, y=30.0, vx=5.0, vy=0.0),
            MotionVector(x=31.0, y=30.0, vx=-5.0, vy=0.0),  # opposite
            MotionVector(x=30.0, y=31.0, vx=0.0, vy=5.0),  # orthogonal
        ]
        clusterer = MotionClusterer(eps=15.0, min_samples=3, velocity_similarity=0.8)
        clusters = clusterer.cluster(vecs)
        # Should not group them because velocity similarity threshold fails
        assert len(clusters) == 0

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="eps"):
            MotionClusterer(eps=0.0)
        with pytest.raises(ValueError, match="min_samples"):
            MotionClusterer(min_samples=0)


class TestMotionAnomalyDetector:
    """Tests for velocity burstiness and sudden acceleration detection."""

    def test_steady_motion_zero_anomaly(self):
        detector = MotionAnomalyDetector(window_size=10, min_history=5)
        # Feed 10 frames with constant speed 2.0
        for _ in range(10):
            score = detector.update(current_speed=2.0, current_direction_rad=0.0)
        assert score == pytest.approx(0.0)

    def test_sudden_burst_spikes_score(self):
        detector = MotionAnomalyDetector(window_size=15, z_threshold=2.0, min_history=5)
        # Baseline: 10 frames with small speed around 1.0
        for _ in range(10):
            detector.update(current_speed=1.0, current_direction_rad=0.0)

        # Sudden massive surge: speed 20.0
        spike_score = detector.update(current_speed=20.0, current_direction_rad=0.0)
        assert spike_score > 0.5

    def test_direction_reversal_penalty(self):
        detector = MotionAnomalyDetector(window_size=10, min_history=5)
        for _ in range(8):
            detector.update(current_speed=5.0, current_direction_rad=0.0)

        # Reversal: direction flips to pi (180 deg) at speed 5.0
        reversal_score = detector.update(current_speed=5.0, current_direction_rad=math.pi)
        assert reversal_score >= 0.3

    def test_reset(self):
        detector = MotionAnomalyDetector()
        detector.update(5.0)
        detector.reset()
        assert len(detector._speed_history) == 0
        assert detector._prev_direction is None

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="window_size"):
            MotionAnomalyDetector(window_size=2)
        with pytest.raises(ValueError, match="z_threshold"):
            MotionAnomalyDetector(z_threshold=-1.0)


class TestMotionField:
    """Tests for MotionField aggregation and reporting."""

    def test_from_vectors_computation(self):
        vecs = [
            MotionVector(x=10.0, y=10.0, vx=3.0, vy=0.0),
            MotionVector(x=20.0, y=20.0, vx=5.0, vy=0.0),
        ]
        field = MotionField.from_vectors(
            vectors=vecs,
            anomaly_score=0.15,
            timestamp=1.2,
            frame_shape=(100, 100),
        )

        assert len(field.velocity_vectors) == 2
        assert field.dominant_speed == pytest.approx(4.0)
        assert field.dominant_direction == pytest.approx(0.0)
        assert field.mean_velocity() == (4.0, 0.0)
        assert field.total_motion_energy() == pytest.approx(9.0 + 25.0)
        assert field.anomaly_score == pytest.approx(0.15)
        assert field.timestamp == pytest.approx(1.2)
        assert field.frame_shape == (100, 100)

        filtered = field.filter_by_magnitude(min_speed=4.0)
        assert len(filtered) == 1
        assert filtered[0].vx == 5.0

    def test_summary_and_serialization(self):
        field = MotionField.from_vectors(vectors=[], anomaly_score=0.0)
        summary = field.summary()
        assert "MotionField" in summary
        assert "vectors=0" in summary

        data = field.to_dict()
        assert data["num_vectors"] == 0
        assert data["anomaly_score"] == 0.0
