"""
test_person_cropper.py — Unit Tests for PersonCropper (Group A)
"""
import pytest
import numpy as np
from vision.crop.person_cropper import PersonCropper, crop_person


def test_cropper_default_padding():
    cropper = PersonCropper()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    bbox = [100, 200, 200, 500]  # bw=100, bh=300

    crop, meta = cropper.crop(frame, bbox)

    assert meta["valid"] is True
    # top_pad = 0.25 * 300 = 75 -> ny1 = 200 - 75 = 125
    # bottom_pad = 0.10 * 300 = 30 -> ny2 = 500 + 30 = 530
    # side_pad = 0.10 * 100 = 10 -> nx1 = 100 - 10 = 90, nx2 = 200 + 10 = 210
    assert meta["padded_bbox"] == [90, 125, 210, 530]
    assert crop.shape == (530 - 125, 210 - 90, 3)


def test_cropper_boundary_clamping():
    cropper = PersonCropper(top_pad=0.50, side_pad=0.50, bottom_pad=0.50)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 90, 90]

    crop, meta = cropper.crop(frame, bbox)

    assert meta["valid"] is True
    assert meta["padded_bbox"][0] == 0  # clamped left
    assert meta["padded_bbox"][1] == 0  # clamped top
    assert meta["padded_bbox"][2] == 100  # clamped right
    assert meta["padded_bbox"][3] == 100  # clamped bottom
    assert crop.shape == (100, 100, 3)


def test_cropper_custom_padding():
    cropper = PersonCropper(top_pad=0.30, bottom_pad=0.15, side_pad=0.05)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    bbox = [100, 100, 200, 300]  # bw=100, bh=200

    crop, meta = cropper.crop(frame, bbox)

    # pad_t = 0.30 * 200 = 60 -> ny1 = 40
    # pad_b = 0.15 * 200 = 30 -> ny2 = 330
    # pad_s = 0.05 * 100 = 5 -> nx1 = 95, nx2 = 205
    assert meta["padded_bbox"] == [95, 40, 205, 330]


def test_cropper_invalid_and_empty_inputs():
    cropper = PersonCropper()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Invalid bbox: x2 <= x1
    crop, meta = cropper.crop(frame, [50, 50, 50, 80])
    assert meta["valid"] is False
    assert crop.size == 0

    # Invalid bbox: y2 <= y1
    crop, meta = cropper.crop(frame, [10, 80, 50, 50])
    assert meta["valid"] is False

    # Invalid frame type
    with pytest.raises(ValueError, match="Invalid frame"):
        cropper.crop(None, [10, 10, 50, 50])

    # Helper function one-off
    crop_h, meta_h = crop_person(frame, [10, 10, 40, 40])
    assert meta_h["valid"] is True
