"""OpenAI text embeddings (text-embedding-3-small, 1536-d).

Only called for words that are not yet in data/embedding/cached_embedding.pkl
(see semantic_similarity.DynamicSemanticEmbedding), so reruns from the cached
embeddings need no API key or network access.
"""

from config import cfg

EMBEDDING_MODEL = "text-embedding-3-small"

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=cfg.OPENAI_API_KEY,
            organization=getattr(cfg, "OPENAI_ORGANIZATION", None),
            project=getattr(cfg, "OPENAI_PROJECT", None),
        )
    return _client


def get_semantic_embedding(text):
    """Embedding vector (list of 1536 floats) of one text."""
    response = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text.strip())
    return response.data[0].embedding


def get_semantic_embedding_list(text_list):
    return [get_semantic_embedding(text) for text in text_list]
