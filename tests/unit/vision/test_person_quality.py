"""
test_person_quality.py — Unit Tests for PersonQualityAssessor (Group B)
"""
import pytest
import numpy as np
from vision.quality.person_quality import PersonQualityAssessor, QualityResult


def test_quality_score_range_and_usability():
    assessor = PersonQualityAssessor(min_usable_threshold=0.40)
    # High resolution sharp synthetic crop
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)

    res = assessor.assess_crop(crop)

    assert isinstance(res, QualityResult)
    assert 0.0 <= res.quality_score <= 1.0
    assert res.is_usable is True
    assert res.blur_score > 0.0
    assert res.resolution_score > 0.0


def test_configurable_blur_threshold():
    # Lower threshold = more lenient blur score
    lenient = PersonQualityAssessor(blur_threshold=10.0)
    strict = PersonQualityAssessor(blur_threshold=100.0)

    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

    res_lenient = lenient.assess_crop(crop)
    res_strict = strict.assess_crop(crop)

    assert res_lenient.blur_score >= res_strict.blur_score


def test_back_facing_person_is_usable():
    """Supervisor Directive 2: Back-facing subjects are USABLE for Re-ID."""
    assessor = PersonQualityAssessor()
    crop = np.random.randint(0, 255, (180, 80, 3), dtype=np.uint8)

    res_front = assessor.assess_crop(crop, orientation="front")
    res_back = assessor.assess_crop(crop, orientation="back")
    res_side = assessor.assess_crop(crop, orientation="side")

    assert res_front.is_usable is True
    assert res_back.is_usable is True
    assert res_side.is_usable is True

    # Orientation score should be 1.0 for all valid views
    assert res_front.orientation_score == 1.0
    assert res_back.orientation_score == 1.0
    assert res_side.orientation_score == 1.0


test_back_facing_person_is_usable()


def test_edge_clipping_penalty():
    assessor = PersonQualityAssessor()
    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

    # Clean unclipped bbox
    res_clean = assessor.assess_crop(
        crop, bbox_in_frame=[100, 100, 150, 200], frame_dimensions=(1000, 1000)
    )
    # Clipped on left edge
    res_clipped = assessor.assess_crop(
        crop, bbox_in_frame=[0, 100, 50, 200], frame_dimensions=(1000, 1000)
    )

    assert res_clean.edge_score > res_clipped.edge_score
    assert "clipped_left" in res_clipped.violations


def test_severe_occlusion_penalty():
    assessor = PersonQualityAssessor()
    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

    res_normal = assessor.assess_crop(crop, is_severely_occluded=False)
    res_occluded = assessor.assess_crop(crop, is_severely_occluded=True)

    assert res_normal.quality_score > res_occluded.quality_score
    assert "severe_occlusion" in res_occluded.violations


def test_empty_crop_returns_zero():
    assessor = PersonQualityAssessor()
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)

    res = assessor.assess_crop(empty_crop)

    assert res.quality_score == 0.0
    assert res.is_usable is False
    assert "empty_crop" in res.violations
