import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from FlagEmbedding import BGEM3FlagModel


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

class BgeM3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)
    
    def create_embedding(self, text):
        return self.model.encode(text, 
                                batch_size=12, 
                                max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    # sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    # sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
            #    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
    # embeddings_2 = model.encode(sentences_2)['dense_vecs']
    # similarity = embeddings_1 @ embeddings_2.T
    # print(similarity)