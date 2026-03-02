from abc import ABC, abstractmethod
import os
from typing import Optional, Union
from dataclasses import asdict
import pandas as pd
from datasets import Dataset
import logging
import torch
import numpy as np

from .config import build_mte_config, MTEConfig, Method
from .utils.common import init_logger
from .utils.chunking import chunk_batch_fn


class BaseMTE(ABC):
    """
    Base class for Mechanistic Topic Extractors (MTEs).
    Provides common functionality and interface for MTE implementations.

    Attributes
    ----------
    cfg : MTEConfig
        Configuration object for the MTE.
    device : torch.device
        Device on which computations will be performed (CPU or GPU).
    hf_token : Optional[str]
        Hugging Face token for accessing private models/datasets. 
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        method: Method,
        config_file: str = "src/sae/config.yaml",
        section: str = "mte",
        **kwargs,
    ):
        self._logger = logger if logger else init_logger(
            config_file=config_file,
            name=__name__
        )

        self.cfg: MTEConfig = build_mte_config(
            logger=logger,
            method=method,
            config_file=config_file,
            section=section,
            **kwargs,
        )

        self._device = self._resolve_device(self.cfg.gen.device)
        self._hf_token = os.getenv(self.cfg.gen.hf_token_env, None)

        try:
            self._logger.info("MTE config: %s", asdict(self.cfg))
        except Exception:
            self._logger.info("MTE config loaded (asdict failed).")

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device("cuda")
        if device == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Invalid device setting: {device!r}")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def hf_token(self) -> Optional[str]:
        return self._hf_token

    def save_dataset(self, dataset: Dataset, path: str, format: str = "json"):
        """Save the dataset to the specified path."""
        if format == "disk":
            dataset.save_to_disk(path)
            self._logger.info(f"Dataset saved to disk at {path}.")
        elif format == "json":
            dataset.to_json(path)
            self._logger.info(f"Dataset saved to JSON at {path}.")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def chunk_corpus(
        self,
        corpus: Dataset,
        text_col: str = "text",
        id_col: str = "id",
        overlap: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Dataset:
        """
        Chunk the corpus into smaller overlapping chunks.
        If do_chunking is False, do NOT chunk: only tokenize and return 1 row per doc (but still produce token_ids so downstream works).

        Parameters
        ----------
        corpus : Dataset
            The dataset containing documents to chunk.
        text_col : str
            The name of the column containing the text.
        id_col : str
            The name of the column containing the document IDs.

        Returns
        -------
        Dataset
            The chunked dataset.
        """

        real_chunk_size = chunk_size if chunk_size is not None else self.cfg.gen.chunk_size
        real_overlap = overlap if overlap is not None else self.cfg.gen.chunk_overlap

        model_limit = self._tok.model_max_length if hasattr(
            self._tok, "model_max_length") else 512

        # if chunking disabled, just tokenize and return 1 row per doc
        if not getattr(self.cfg.gen, "do_chunking", True):
            def tok_only(batch, tokenizer, text_col, id_col, max_len):
                toks = tokenizer(
                    batch[text_col],
                    truncation=True,  # @TODO: check if alternative
                    max_length=max_len,
                    add_special_tokens=True,
                )
                return {
                    "doc_id": batch[id_col],
                    "text": batch[text_col],
                    "token_ids": toks["input_ids"],
                }

            max_len = getattr(self.cfg.llm, "max_seq_len", None)
            if max_len is None:
                max_len = min(
                    int(getattr(self._tok, "model_max_length", 2048)), 4096)

            self._logger.info(
                f"Chunking disabled. Tokenizing with truncation to max_len={max_len} tokens.")

            return corpus.map(
                tok_only,
                fn_kwargs={"tokenizer": self._tok, "text_col": text_col,
                           "id_col": id_col, "max_len": max_len},
                batched=True,
                batch_size=self.cfg.gen.chunk_batch_size,
                num_proc=self.cfg.gen.chunk_num_proc,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        # if chunking enabled, adjust chunk size if exceeds model limit
        if getattr(self.cfg.gen, "adjust_chunk_size", False) and (real_chunk_size is not None) and (real_chunk_size > model_limit):
            self._logger.warning(
                f"Chunk size {real_chunk_size} exceeds model limit {model_limit}. Adjusting chunk size."
            )
            real_chunk_size = model_limit

            if real_overlap is not None and real_overlap >= real_chunk_size:
                real_overlap = int(real_chunk_size * 0.25)
                self._logger.warning(
                    f"Chunk overlap adjusted to {real_overlap}.")

        self._logger.info(
            f"Chunking corpus with chunk_size={real_chunk_size}, chunk_overlap={real_overlap}..."
        )

        chunked_dataset = corpus.map(
            chunk_batch_fn,
            fn_kwargs={
                "tokenizer": self._tok,
                "chunk_size": real_chunk_size,
                "chunk_overlap": real_overlap,
                "text_col": text_col,
                "id_col": id_col,
            },
            batched=True,
            batch_size=self.cfg.gen.chunk_batch_size,
            num_proc=self.cfg.gen.chunk_num_proc,
            remove_columns=corpus.column_names,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return chunked_dataset

    def _aggregate_chunk_saes(
        self,
        activationes_chunks: Dataset,
    ) -> Dataset:
        """
        Aggregate chunk-level SAE activations back to document-level.

        Parameters
        ----------
        activationes_chunks : Dataset
            Dataset containing chunk-level SAE activations.

        Returns
        -------
        Dataset
            Dataset with document-level SAE activations.
        """

        df = activationes_chunks.to_pandas()

        if self.cfg.gen.doc_agg == "mean":
            agg_df = df.groupby("doc_id")["theta"].apply(
                lambda x: np.mean(np.stack(x), axis=0)
            ).reset_index()
        elif self.cfg.gen.doc_agg == "max":
            agg_df = df.groupby("doc_id")["theta"].apply(
                lambda x: np.max(np.stack(x), axis=0)
            ).reset_index()
        else:
            raise ValueError(
                f"Unsupported aggregation method: {self.cfg.gen.doc_agg}")

        return Dataset.from_pandas(agg_df)

    def normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Normalize theta so that each row sums to 1."""

        theta = theta / theta.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @abstractmethod
    def fit_transform(
        self,
        data: Union[Dataset, pd.DataFrame],
        text_col: str = "text",
        id_col: str = "doc_id",
        return_chunks: bool = False,
    ):
        """Return doc-level outputs (and optionally chunk-level)."""
        ...
