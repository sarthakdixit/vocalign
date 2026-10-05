import numpy as np
import pytest

from core.infer import similarity


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert similarity.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert similarity.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert similarity.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_handles_zero_vector_without_crashing():
    assert similarity.cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_reference_centroid_is_unit_normalized():
    centroid = similarity.reference_centroid([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert np.linalg.norm(centroid) == pytest.approx(1.0)
    assert centroid[0] == pytest.approx(centroid[1])


def test_reference_centroid_single_embedding_normalizes_it():
    centroid = similarity.reference_centroid([np.array([3.0, 4.0])])  # norm 5
    assert centroid[0] == pytest.approx(0.6)
    assert centroid[1] == pytest.approx(0.8)


def test_reference_centroid_handles_zero_vector_without_crashing():
    centroid = similarity.reference_centroid([np.array([0.0, 0.0])])
    np.testing.assert_array_equal(centroid, np.array([0.0, 0.0]))


def test_compute_embedding_uses_injected_preprocess_then_encoder():
    seen = {}

    def fake_preprocess(samples, sample_rate):
        seen["samples"] = samples
        seen["sample_rate"] = sample_rate
        return "preprocessed-wav"

    class FakeEncoder:
        def embed_utterance(self, wav):
            seen["wav_passed_to_encoder"] = wav
            return np.array([0.1, 0.2, 0.3])

    samples = np.zeros(10, dtype=np.float32)
    result = similarity.compute_embedding(samples, 16000, encoder=FakeEncoder(), preprocess_fn=fake_preprocess)

    assert seen["sample_rate"] == 16000
    assert seen["wav_passed_to_encoder"] == "preprocessed-wav"
    np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
