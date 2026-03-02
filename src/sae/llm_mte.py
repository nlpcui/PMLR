import logging
import os
from contextlib import nullcontext
from typing import Literal, Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from src.sae.base import BaseMTE
from src.sae.utils.neuronpedia_api import prefetch_feature_explanations
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

SAEBackend = Literal["sae_lens", "sparsify"]


def _ensure_pad(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[PreTrainedModel] = None,
):
    """
    Ensure that the tokenizer has a pad_token, and that the model's
    config.pad_token_id is set accordingly.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token_id to fallback to.")
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _load_sae_lens(release: str, sae_id: str, device):
    from sae_lens import SAE  # type: ignore
    sae, _, _ = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device)
    return sae


def _load_sparsify(release: str, sae_id: str, device):
    """
    Load a SAE from a sparsify/EleutherAI HuggingFace repo.

    Parameters
    ----------
    release : str
        HuggingFace repo id, e.g. "EleutherAI/sae-llama-3-8b-32x".
    sae_id : str
        Hookpoint name used by sparsify, e.g. "layers.15".
    """
    from sparsify import Sae  # type: ignore
    return Sae.load_from_hub(release, hookpoint=sae_id).to(device)


def _encode_sae_lens(sae, resid_flat: torch.Tensor) -> torch.Tensor:
    """Returns a dense [N, F] activation tensor."""
    return sae.encode(resid_flat)


def _encode_sparsify(sae, resid_flat: torch.Tensor) -> torch.Tensor:
    """
    sparsify returns a SparseActs namedtuple (.top_acts [N,k], .top_indices [N,k]).
    We scatter into a dense [N, F] tensor so the rest of the pipeline is unchanged.
    """
    acts = sae.encode(resid_flat)
    N = resid_flat.shape[0]
    F = sae.num_latents
    dense = torch.zeros(N, F, dtype=resid_flat.dtype, device=resid_flat.device)
    dense.scatter_(1, acts.top_indices.long(), acts.top_acts)
    return dense


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMBasedMTE(BaseMTE):
    def __init__(
        self,
        logger: logging.Logger = None,
        sae_backend: SAEBackend = "sae_lens",
        **kwargs,
    ):
        """
        Parameters
        ----------
        sae_backend : {"sae_lens", "sparsify"}
            Which library to use for loading and running the SAE.
            - "sae_lens"  : original behaviour, uses sae_lens.SAE (dense output).
            - "sparsify"  : uses EleutherAI's sparsify.Sae (TopK sparse, converted
                            to dense internally so the rest of the pipeline is unchanged).
        """
        super().__init__(logger=logger, method="llm", **kwargs)
        assert self.cfg.gen is not None, "GeneralConfig missing"
        assert self.cfg.llm is not None, "LLMConfig missing"

        self._sae_backend: SAEBackend = sae_backend

        llm_model = self.cfg.llm.llm_model
        self._tok = AutoTokenizer.from_pretrained(
            llm_model, token=self.hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            token=self.hf_token,
            torch_dtype=(torch.bfloat16 if self._device.type ==
                         "cuda" else torch.float32),
        ).to(self._device)

        if sae_backend == "sae_lens":
            self._sae = _load_sae_lens(
                self.cfg.llm.sae_release, self.cfg.llm.sae_id, self._device)
            self._encode = _encode_sae_lens
        elif sae_backend == "sparsify":
            self._sae = _load_sparsify(
                self.cfg.llm.sae_release, self.cfg.llm.sae_id, self._device)
            self._encode = _encode_sparsify
        else:
            raise ValueError(
                f"Unknown sae_backend: {sae_backend!r}. Choose 'sae_lens' or 'sparsify'.")

        _ensure_pad(self._tok, self._model)

    def chunk_corpus(
        self,
        corpus: Dataset,
        text_col: str = "text",
        id_col: str = "id",
    ) -> Dataset:
        """
        Chunk the corpus into smaller overlapping chunks. Calls the chunking
        method from BaseMTE with appropriate chunk size. If chunking is
        disabled, we still call the method with effectively no chunking and
        overlap 0 so we get the token_ids column.
        """
        if self.cfg.gen.chunk_size is None:
            model_max = getattr(self._tok, "model_max_length", 5000)
            if model_max is None or model_max > 100_000:
                model_max = 5000
            chunk_size = model_max
        else:
            chunk_size = self.cfg.gen.chunk_size

        overlap = self.cfg.gen.chunk_overlap if self.cfg.gen.do_chunking else 0

        return super().chunk_corpus(
            corpus,
            text_col=text_col,
            id_col=id_col,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def _aggregate_feats(
        self,
        feats: torch.Tensor,
        attn_mask: torch.Tensor,
        agg: str,
        threshold: Optional[float],
    ) -> torch.Tensor:
        mask = attn_mask.bool()
        if agg == "mean":
            feats_masked = feats * mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return feats_masked.sum(dim=1) / denom
        if agg == "max":
            feats_for_max = feats.masked_fill(
                ~mask.unsqueeze(-1), float("-inf"))
            out = feats_for_max.max(dim=1).values
            return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        if agg == "frac_above":
            if threshold is None:
                raise ValueError("threshold required for agg='frac_above'")
            above = (feats > threshold) & mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return above.float().sum(dim=1) / denom
        raise ValueError(f"Unknown agg: {agg}")

    def _add_chunk_theta_all_features(self, batch):
        token_id_lists = batch.get("token_ids", [])
        if not token_id_lists:
            return {"theta": [], "num_features": [], "theta_sparse": []}

        padded = self._tok.pad(
            [{"input_ids": ids} for ids in token_id_lists],
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=(8 if self._device.type == "cuda" else None),
        )
        input_ids = padded["input_ids"].to(self._device)
        attn_mask = padded["attention_mask"].to(self._device)

        use_cuda = self._device.type == "cuda"
        amp_ctx = torch.autocast(
            "cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

        with torch.inference_mode(), amp_ctx:
            out = self._model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            hs = out.hidden_states[self.cfg.llm.layer_index]   # [B, T, D]
            B, T, D = hs.shape
            resid_flat = hs.reshape(B * T, D)

            feats_flat = self._encode(self._sae, resid_flat)    # [B*T, F]
            feats = feats_flat.reshape(B, T, -1)                # [B, T, F]

            theta = self._aggregate_feats(
                feats, attn_mask,
                agg=self.cfg.llm.chunk_agg,
                threshold=self.cfg.llm.threshold,
            )   # [B, F]

            theta_sparse = self._build_theta_sparse(
                feats=feats,
                input_ids=input_ids,
                attn_mask=attn_mask,
                theta=theta,
                top_tokens_per_feature=self.cfg.llm.top_tokens_per_feature,
                act_threshold=0.0,
                theta_threshold=None,
            )

        theta_list = theta.float().cpu().tolist()
        F = theta.shape[-1]

        return {
            "theta": theta_list,
            "theta_sparse": theta_sparse,
            "num_features": [F] * len(theta_list),
        }

    def _build_theta_sparse(
        self,
        feats: torch.Tensor,        # [B, T, F]
        input_ids: torch.Tensor,    # [B, T]
        attn_mask: torch.Tensor,    # [B, T]
        theta: torch.Tensor,        # [B, F]
        *,
        top_tokens_per_feature: int = 1,
        act_threshold: float = 0.0,
        theta_threshold: Optional[float] = None,
    ):
        """
        For each active feature, find the top-k tokens that activated it most.

        Each entry per batch element is a dict with:
            feature_id        : int
            feature_weight    : float
            token_pos         : List[int]
            token_id          : List[int]
            token_activation  : List[float]
        """
        B, T, F = feats.shape
        mask = attn_mask.bool()
        feats_f = feats.float()
        theta_f = theta.float()

        feats_masked = feats_f.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_act, _ = feats_masked.max(dim=1)    # [B, F]
        active = torch.isfinite(max_act) & (max_act > act_threshold)
        if theta_threshold is not None:
            active = active & (theta_f > theta_threshold)

        out_batch = []
        for b in range(B):
            entries = []
            active_fs = active[b].nonzero(as_tuple=False).squeeze(-1)
            ids_b = input_ids[b]
            valid_len = int(mask[b].sum().item())
            k_tok = min(top_tokens_per_feature, valid_len)

            if k_tok <= 0 or active_fs.numel() == 0:
                out_batch.append(entries)
                continue

            for f in active_fs.tolist():
                a = feats_masked[b, :, f]
                vals, pos = torch.topk(a, k=k_tok, dim=0)
                keep = torch.isfinite(vals)
                if not keep.any():
                    continue
                pos = pos[keep]
                vals = vals[keep]
                tok_ids = ids_b[pos].to(torch.int64).tolist()
                tok_str = self._tok.convert_ids_to_tokens(tok_ids)
                entries.append({
                    "feature_id": int(f),
                    "feature_weight": float(theta_f[b, f].item()),
                    "token_pos": pos.to(torch.int64).tolist(),
                    "token_id": tok_ids,
                    "token_str": tok_str,
                    "token_activation": vals.tolist(),
                })
            out_batch.append(entries)

        return out_batch

    def _aggregate_chunk_saes(self, activationes_chunks: Dataset) -> Dataset:
        """
        Document-level aggregation of chunk-level SAEs. The BaseMTE version of this method aggregates the dense "theta" features and counts "num_features". We override it to also concatenate the "theta_sparse" lists across chunks for each document.
        Uses the parent method to aggregate the normal features, then
        concatenates the theta_sparse lists per document.
        """
        doc_level = super()._aggregate_chunk_saes(activationes_chunks)

        if "theta_sparse" not in activationes_chunks.column_names:
            return doc_level

        from itertools import groupby as _groupby

        # Sort by doc_id and concatenate theta_sparse lists — no pandas needed,
        # so there is no risk of numpy coercion mangling the list-of-dicts.
        rows = sorted(
            zip(activationes_chunks["doc_id"],
                activationes_chunks["theta_sparse"]),
            key=lambda t: t[0],
        )
        sparse_by_doc = {
            doc_id: [entry for _, chunk in chunks for entry in chunk]
            for doc_id, chunks in _groupby(rows, key=lambda t: t[0])
        }

        doc_df = doc_level.to_pandas()
        doc_df["theta_sparse"] = doc_df["doc_id"].map(
            lambda did: sparse_by_doc.get(did, [])
        )
        return Dataset.from_pandas(doc_df)

    def _enrich_with_neuronpedia(self, doc_level: Dataset) -> Dataset:
        """
        Pre-fetch Neuronpedia explanations for every unique feature_id present
        in theta_sparse (in parallel), then apply labels in a single fast map.

        Uses cfg.llm.neuronpedia_{api_base,model_id,sae_id}.
        Features that cannot be fetched get the label "<fetch error>".
        """
        if "theta_sparse" not in doc_level.column_names:
            return doc_level

        api_base = self.cfg.llm.neuronpedia_api_base
        model_id = self.cfg.llm.neuronpedia_model_id
        sae_id = self.cfg.llm.neuronpedia_sae_id

        # 1. Collect all unique feature ids across the whole dataset
        all_fids = [
            entry["feature_id"]
            for row in doc_level["theta_sparse"]
            for entry in row
        ]
        unique_fids = list(set(all_fids))
        self._logger.info(
            "Pre-fetching %d unique Neuronpedia features (model=%s, sae=%s) …",
            len(unique_fids), model_id, sae_id,
        )

        # 2. Bulk-fetch in parallel — no HTTP inside map()
        expl_cache = prefetch_feature_explanations(
            feature_ids=unique_fids,
            neronepedia_api_base=api_base,
            neronepedia_model_id=model_id,
            neronepedia_sae_id=sae_id,
        )
        self._logger.info("Neuronpedia pre-fetch done.")

        # 3. Apply labels locally — pure dict lookup, no I/O
        def _enrich_row(example):
            enriched = [
                {**entry,
                    "feature_label": expl_cache.get(entry["feature_id"], "<fetch error>")}
                for entry in example["theta_sparse"]
            ]
            return {"theta_sparse": enriched}

        return doc_level.map(_enrich_row, desc="neuronpedia enrichment")

    def fit_transform(
        self,
        data: Union[Dataset, pd.DataFrame],
        text_col: str = "text",
        id_col: str = "id",
        save_chunks: bool = False,
        save_path: str = "./",
        *,
        batch_size: int = 8,
        enrich_neuronpedia: bool = True,
    ):
        """
        Parameters
        ----------
        enrich_neuronpedia : bool
            If True (default), fetch Neuronpedia explanations for every active
            feature and add ``feature_label`` to each theta_sparse entry.
            Set to False to skip network calls (faster / offline).
        """
        if isinstance(data, pd.DataFrame):
            data = Dataset.from_pandas(data)

        chunks = self.chunk_corpus(data, text_col=text_col, id_col=id_col)
        chunks = chunks.map(
            self._add_chunk_theta_all_features,
            batched=True,
            batch_size=batch_size,
        )

        doc_level = (
            self._aggregate_chunk_saes(chunks)
            if self.cfg.gen.do_chunking
            else chunks
        )

        if enrich_neuronpedia:
            doc_level = self._enrich_with_neuronpedia(doc_level)

        os.makedirs(save_path, exist_ok=True)
        chunk_fp = os.path.join(save_path, "llm_chunk_level_dataset.json")
        doc_fp = os.path.join(save_path, "llm_document_level_dataset.json")

        for fp in [doc_fp, chunk_fp]:
            if os.path.exists(fp):
                os.remove(fp)

        if save_chunks:
            self.save_dataset(chunks, path=chunk_fp, format="json")
        self.save_dataset(doc_level, path=doc_fp, format="json")
