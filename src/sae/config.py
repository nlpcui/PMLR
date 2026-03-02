from __future__ import annotations
from .utils.common import from_dict_strict, merge_dicts
from .utils.common import load_yaml_config_file
from dataclasses import dataclass, field
from typing import Optional, Literal

Method = Literal["emb", "llm"]
Agg = Literal["mean", "max", "frac_above"]


@dataclass(frozen=True)
class GeneralConfig:
    hf_token_env: str = "HUGGINGFACE_TOKEN"
    do_chunking: bool = False
    chunk_size: int = 128
    chunk_overlap: int = 32
    chunk_batch_size: int = 200
    chunk_num_proc: int = 4
    embed_batch_size: int = 32
    adjust_chunk_size: bool = True
    doc_agg: Agg = "max"
    device : str = "cuda"


@dataclass(frozen=True)
class EmbeddingsConfig:
    embeddings_model: str

@dataclass(frozen=True)
class LLMConfig:
    llm_model: str
    layer_index: int
    sae_release: str
    sae_id: str

    neuronpedia_api_base: str = "https://www.neuronpedia.org/api/feature"
    neuronpedia_model_id: str = "gemma-2-2b"
    neuronpedia_sae_id: str = "24-gemmascope-res-16k"
    chunk_agg: Agg = "max"
    top_k: int = 20
    top_tokens_per_feature: int = 20
    M: int = 256
    K: int = 8
    threshold: float = 0.5
    pad_to_multiple_of: Optional[int] = 8


@dataclass(frozen=True)
class MTEConfig:
    method: Method
    gen: GeneralConfig = field(default_factory=GeneralConfig)
    emb: Optional[EmbeddingsConfig] = None
    llm: Optional[LLMConfig] = None


def build_mte_config(
    *,
    logger,
    method: str,
    config_file: str = ".config.yaml",
    section: str = "mte",
    **kwargs,
) -> MTEConfig:
    raw = load_yaml_config_file(
        config_file=config_file, section=section, logger=logger)

    general_raw = raw.get("gen", {})
    method_raw = raw.get(method, {})

    # override with kwargs
    merged = merge_dicts(general_raw, method_raw, kwargs)

    # construct first with general config and the children-specific
    general_keys = {k: v for k, v in merged.items(
    ) if k in {f.name for f in GeneralConfig.__dataclass_fields__.values()}}
    general = from_dict_strict(GeneralConfig, general_keys)

    if method == "emb":
        emb_keys = {k: v for k, v in merged.items(
        ) if k in EmbeddingsConfig.__dataclass_fields__}
        embeddings = from_dict_strict(EmbeddingsConfig, emb_keys)
        return MTEConfig(method="emb", gen=general, emb=embeddings, llm=None)

    if method == "llm":
        llm_keys = {k: v for k, v in merged.items(
        ) if k in LLMConfig.__dataclass_fields__}
        llm = from_dict_strict(LLMConfig, llm_keys)
        return MTEConfig(method="llm", gen=general, emb=None, llm=llm)

    raise ValueError(f"Unknown method: {method}")
