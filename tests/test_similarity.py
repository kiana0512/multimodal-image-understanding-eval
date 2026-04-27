import numpy as np
from mm_eval.retrieval.similarity import cosine_similarity, topk_indices, pairwise_cosine_similarity

def test_cosine_similarity_identity():
    assert cosine_similarity(np.array([1,0]), np.array([1,0])) == 1.0

def test_cosine_similarity_orthogonal():
    assert abs(cosine_similarity(np.array([1,0]), np.array([0,1]))) < 1e-6

def test_topk_indices_vector():
    scores=np.array([0.1,0.9,0.2])
    assert topk_indices(scores,k=2).tolist() == [1,2]

def test_pairwise_shape():
    a=np.eye(2); b=np.eye(2)
    assert pairwise_cosine_similarity(a,b).shape == (2,2)
