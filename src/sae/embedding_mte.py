import logging
import os
import pathlib
from typing import Union, Optional
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


from .base import BaseMTE
from .sae import SparseAutoencoder, load_model, get_sae_checkpoint_name


class EmbeddingBasedMTE(BaseMTE):
    def __init__(self, logger: logging.Logger = None, **kwargs):
        super().__init__(logger=logger, method="emb", **kwargs)

        assert self.cfg.gen is not None, "GeneralConfig missing"
        assert self.cfg.emb is not None, "EmbeddingsConfig missing"

        self._st = SentenceTransformer(self.cfg.emb.embeddings_model)
        self._tok = AutoTokenizer.from_pretrained(
            self._st.tokenizer.name_or_path
        )

    def _embed_chunks(self, dataset: Dataset, text_col: str) -> Dataset:
        """
        Embed the chunks in the dataset using the SentenceTransformer model.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing text chunks to embed.
        text_col : str
            The name of the column containing the text to embed.
        batch_size : int
            The batch size to use for embedding.
        """
        if self._st is None:
            raise ValueError("SentenceTransformer model is not initialized.")

        self._logger.info("Embedding dataset...")

        def embed_batch(batch):
            embs = self._st.encode(
                batch[text_col],
                batch_size=self.cfg.gen.embed_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return {"embeddings": embs}

        self._logger.info("Embedding chunks...")

        return dataset.map(embed_batch, batched=True, batch_size=self.cfg.gen.embed_batch_size)

    def _train_sae(
        self,
        embeddings: np.ndarray,
        M: int,
        K: int,
        *,
        checkpoint_dir: Optional[str] = None,
        overwrite_checkpoint: bool = False,
        val_embeddings: Optional[np.ndarray] = None,
        **train_kwargs,
    ) -> SparseAutoencoder:
        """
        Train a Sparse Autoencoder or load an existing one.

        Original source: https://github.com/rmovva/HypotheSAEs/blob/main/hypothesaes/sae.py
        Modifications: Minor edits made to integrate logging consistent with this module.
        """

        embeddings = np.asarray(embeddings)
        input_dim = embeddings.shape[1]

        X = torch.tensor(embeddings, dtype=torch.float)
        X_val = torch.tensor(
            val_embeddings, dtype=torch.float) if val_embeddings is not None else None

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = get_sae_checkpoint_name(M, K)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(checkpoint_path) and not overwrite_checkpoint:
                return load_model(checkpoint_path)

        sae = SparseAutoencoder(
            input_dim=input_dim, m_total_neurons=M, k_active_neurons=K)

        sae.fit(X_train=X, X_val=X_val, save_dir=checkpoint_dir, **train_kwargs)

        return sae

    def _get_chunk_emb_saes(
        self,
        embedded_chunks: Dataset,
        M: int,
        K: int,
        test_size: float = 0.12,
        random_state: int = 42,
        checkpoint_dir: str = "./checkpoints",
        cache_name: str = "mte_sae_cache",
        **train_kwargs,
    ) -> Dataset:
        """
        Train SAE on chunk embeddings and obtain sparse activations.
        """

        embeddings = embedded_chunks["embeddings"]

        train_indices, val_indices = train_test_split(
            range(len(embeddings)), test_size=test_size, random_state=random_state
        )

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)
        train_e = [embeddings[i] for i in train_indices]
        val_e = [embeddings[i] for i in val_indices]

        checkpoint_dir = pathlib.Path(checkpoint_dir) / cache_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model = self._train_sae(
            embeddings=train_e,
            M=M,
            K=K,
            checkpoint_dir=pathlib.Path(checkpoint_dir) / cache_name,
            val_embeddings=val_e,
            **train_kwargs
        )

        train_activations = model.get_activations(train_e)
        val_activations = model.get_activations(val_e)

        acts = np.asarray(train_activations)
        nonzero = acts[acts > 0]

        # zip train_activations with train_indices to restore original order
        # and same for val_activations
        train_activations = [(train_indices[i], train_activations[i])
                             for i in range(len(train_activations))]
        val_activations = [(val_indices[i], val_activations[i])
                           for i in range(len(val_activations))]
        # concatenate and sort by original index
        all_activations = train_activations + val_activations
        all_activations = sorted(all_activations, key=lambda x: x[0])
        # extract only activations
        activations = [x[1] for x in all_activations]

        # save back to dataset
        augmented_chunks = embedded_chunks.add_column(
            "theta", activations)

        return augmented_chunks

    def fit_transform(
        self,
        data: Union[Dataset, pd.DataFrame],
        text_col: str = "text",
        id_col: str = "id",
        save_chunks: bool = False,
        save_path: str = "./",
        return_chunks: bool = False,
        *,
        M: int = 256,
        K: int = 8,
        **sae_kwargs,
    ):
        if isinstance(data, pd.DataFrame):
            data = Dataset.from_pandas(data)

        if self.cfg.gen.do_chunking:
            chunks = self.chunk_corpus(data, text_col=text_col, id_col=id_col)
        else:
            chunks = data

        chunks = self._embed_chunks(chunks, text_col=text_col)
        chunks = self._get_chunk_emb_saes(chunks, M=M, K=K, **sae_kwargs)

        docs = self._aggregate_chunk_saes(
            chunks) if self.cfg.gen.do_chunking else chunks

        os.makedirs(save_path, exist_ok=True)
        if save_chunks:
            self.save_dataset(chunks, path=os.path.join(
                save_path, "emb_chunk_level_dataset.json"), format="json")
        self.save_dataset(docs, path=os.path.join(
            save_path, "emb_document_level_dataset.json"), format="json")
        
        return (chunks, docs) if return_chunks and self.cfg.gen.do_chunking else docs
