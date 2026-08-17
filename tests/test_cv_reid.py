import numpy as np
from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.quality import CropQualitySelector

def test_quality_selector():
    selector = CropQualitySelector()
    
    # 1. Reject empty crop
    res = selector.assess_quality(np.array([]))
    assert res["approved"] is False
    assert res["reason"] == "empty_crop"
    
    # 2. Reject too small
    small_crop = np.zeros((64, 32, 3), dtype=np.uint8)  # h=64, w=32 < 64
    res = selector.assess_quality(small_crop)
    assert res["approved"] is False
    assert res["reason"] == "too_small_32x64"
    
    # 3. Reject too blurry (Laplacian variance < 100)
    # A completely uniform image has 0 variance
    uniform_crop = np.ones((256, 128, 3), dtype=np.uint8) * 128
    res = selector.assess_quality(uniform_crop)
    assert res["approved"] is False
    assert res["reason"] == "too_blurry"
    
    # 4. Accept good crop (random noise usually has high variance and average brightness)
    np.random.seed(42)
    good_crop = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)
    res = selector.assess_quality(good_crop)
    assert res["approved"] is True

def test_osnet_deterministic_extraction():
    extractor = OSNetExtractor()
    
    # Generate a random crop
    np.random.seed(42)
    crop1 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    crop2 = np.copy(crop1)
    
    vec1 = extractor.extract(crop1)
    vec2 = extractor.extract(crop2)
    
    # Should be deterministic
    assert vec1 == vec2
    assert len(vec1) == 512
    
    # L2 Norm should be 1.0 (cosine similarity requirement)
    norm = np.linalg.norm(vec1)
    assert np.isclose(norm, 1.0)
    
    # Different crop should have different vector
    crop3 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    vec3 = extractor.extract(crop3)
    assert vec1 != vec3
