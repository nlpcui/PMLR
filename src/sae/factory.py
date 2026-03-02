import os
from .embedding_mte import EmbeddingBasedMTE
from .llm_mte import LLMBasedMTE

def create_mte(method: str, **kwargs):
    method = method.lower()
    if method == "emb":
        return EmbeddingBasedMTE(**kwargs)
    if method == "llm":
        hf_token = kwargs.pop("hf_token", None) or os.getenv("HUGGINGFACE_TOKEN")
        return LLMBasedMTE(hf_token=hf_token, **kwargs)
    raise ValueError(f"Unknown method: {method}")
