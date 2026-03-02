from datasets import Dataset


def extract_abstract_from_wiki(corpus_base: Dataset, id_col: str, text_col: str) -> Dataset:
    """Extract the abstract from a Wikipedia dataset and return a new Dataset with only the abstracts (keeping the id column)."""
    
    def get_abstract(example):
        text = example[text_col]
        abstract = text.split("= =")[0].strip()
        return {text_col: abstract}

    corpus_abstracts = corpus_base.map(get_abstract, remove_columns=[col for col in corpus_base.column_names if col != text_col and col != id_col])
    return corpus_abstracts
    
    