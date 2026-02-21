import os

local_config = {
    'data': {
        'topic_wikitext': 'data/topic_wikitext/train.metadata.jsonl',
        'topic_bill': 'data/topic_bill/train.metadata.jsonl',
        'sae_wikitext': 'data/sae_wikitext/sae_wikitext_train.jsonl'
    },
}

openai_key = os.getenv("OPENAI_API_KEY_Data_Interpretation")

api_price = {
    'gpt-4o-mini': {'input': 0.15 / 1000000, 'output': 0.6 / 1000000}
}