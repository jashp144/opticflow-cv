"""Integration tests for OpticalFlowPipeline and public API contracts."""

from __future__ import annotations

import pytest

from opticflow_cv import MotionField, OpticalFlowPipeline
from opticflow_cv.core.config import OpticalFlowConfig
from opticflow_cv.core.models import Frame


class TestOpticalFlowPipelineContracts:
    """Verifies interface contracts specified in PROJECT.md and DISPATCH.md."""

    def test_project_contract_lucas_kanade(self, synthetic_translating_pair):
        frame_prev, frame_curr = synthetic_translating_pair
        pipeline = OpticalFlowPipeline(method="lucas_kanade")
        motion: MotionField = pipeline.process_frames(frame_prev, frame_curr)
        assert motion.velocity_vectors is not None
        assert isinstance(motion.velocity_vectors, list)

    def test_pipeline_farneback_method(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        pipeline = OpticalFlowPipeline(method="farneback")
        motion = pipeline.process_frames(f1, f2)
        assert motion.velocity_vectors is not None
        assert motion.flow_field is not None
        assert motion.flow_field.shape == f1.shape

    def test_pipeline_numpy_gradient_method(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        pipeline = OpticalFlowPipeline(method="numpy_gradient")
        motion = pipeline.process_frames(f1, f2)
        assert motion.velocity_vectors is not None
        assert motion.flow_field is not None

    def test_streaming_process_mode(self, frame_sequence):
        pipeline = OpticalFlowPipeline(method="lucas_kanade")

        # First frame initializes buffer
        m0 = pipeline.process(frame_sequence[0])
        assert len(m0.velocity_vectors) == 0

        # Subsequent frames produce flow
        m1 = pipeline.process(frame_sequence[1])
        assert isinstance(m1, MotionField)
        assert m1.velocity_vectors is not None

        m2 = pipeline.process(frame_sequence[2])
        assert isinstance(m2, MotionField)

        # Check reset
        pipeline.reset()
        m_after_reset = pipeline.process(frame_sequence[0])
        assert len(m_after_reset.velocity_vectors) == 0

    def test_custom_config_injection(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        cfg = OpticalFlowConfig(method="farneback", winsize=21, motion_threshold=1.0)
        pipeline = OpticalFlowPipeline(config=cfg)
        assert pipeline.config.winsize == 21
        motion = pipeline.process_frames(f1, f2)
        assert motion.velocity_vectors is not None

    def test_invalid_method_error(self):
        with pytest.raises(ValueError, match="Invalid optical flow method"):
            OpticalFlowPipeline(method="non_existent")

    def test_frame_object_inputs(self, synthetic_translating_pair):
        f1, f2 = synthetic_translating_pair
        frame1 = Frame.from_array(f1, frame_id=1, timestamp=0.0)
        frame2 = Frame.from_array(f2, frame_id=2, timestamp=0.033)

        pipeline = OpticalFlowPipeline(method="farneback")
        motion = pipeline.process_frames(frame1, frame2)
        assert motion.frame_shape == (120, 160)
