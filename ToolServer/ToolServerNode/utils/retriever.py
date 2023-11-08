import os
import re
import json
import tqdm
import requests
import numpy as np
import logging

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG

logger = logging.getLogger(CONFIG['logger'])
STANDARDIZING_PATTERN = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")

def standardizing(string):
    string = STANDARDIZING_PATTERN.sub("_", string)
    string = re.sub(r"(_)\1+","_", string)
    string = string.strip("_").lower()
    return string

def ada_retriever(doc_embeddings: list, id2tool:dict, question: str, top_k: int=5):
    cfg = CONFIG['retriver']
    url = cfg['endpoint']
    headers = cfg['headers']
    payload = {'input':question}
    payload.update(cfg['payload'])

    response = requests.post(url, json=payload, headers=headers)
    query_embedding = np.array(response.json()['data'][0]['embedding'])

    similarities = cosine_similarity([query_embedding], doc_embeddings)


    sorted_doc_indices = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i], reverse=True)
    return list(
        map(lambda doc_id: id2tool[str(doc_id)], sorted_doc_indices[:top_k])
    )

def build_tool_embeddings(tools_json:list[dict]):
    cfg = CONFIG['retriver']
    if os.path.exists(cfg['id2tool_file']) and os.path.exists(cfg['embedding_file']):
        id2tool = json.load(open(cfg['id2tool_file'], "r"))
        doc_embedings = np.load(cfg['embedding_file'])
        if len(id2tool) != len(doc_embedings):
            logger.error('Embedding file and id2tool file do not match! Rebuild embeddings!')
            id2tool = {}
            doc_embedings = []
    else:
        id2tool = {}
        doc_embedings = []

    # check embedding file whether need to be updated
    # get all current tool names
    # tool_names = set(map(lambda tool_json: tool_json['name'], tools_json))
    # cached_tool_names = set(id2tool.values())
    # if tool_names == cached_tool_names:
    #     logger.info('No tools change, use cached embeddings!')
    #     return doc_embedings, id2tool
    return doc_embedings, id2tool
    
    