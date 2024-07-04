import os
import csv
import pandas as pd
import numpy as np
import tqdm
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from pinecone import Pinecone
from pinecone import ServerlessSpec

# load env
load_dotenv()


TIP_FILE = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_tips_dense.csv"
REVIEW_FILE = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_reviews_dense.csv"

class YelpExpert():

    def __init__(self):
        # self.index = self.init_pinecone()

        # Constants
        self.top_k = 10

        self.tip_summary = self.get_summary(TIP_FILE)
        self.review_summary = self.get_summary(REVIEW_FILE)
        
        self.embedding, self.embedding_shape = self.create_embed(self.tip_summary, self.review_summary)
        # self.index = self.init_pinecone(self.embedding_shape)
        # self.vector_store = self.create_vector(self.embedding, self.embedding_shape)

        self.conversation_history = []
        self.conversation_tokens = 0

    def init_pinecone(self, shape):
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = 'yelp-expert'
        print(shape)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=shape[1],
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )

        return pc.Index(index_name)

    
    def get_summary(self, file_path):
        output = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # print(row)
                output.append(row[0])
        return output

    def create_embed(self, tip_summary, review_summary):
        '''create vector store where each entry is a line of review or tip'''
        print("Creating embedding...")
        split_data = [Document(page_content=x) for x in tip_summary]

        print("length of split data: ",len(split_data), " length of each chunk: ", len(split_data[0].page_content))
        
        split_data = split_data[:100]
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        embeddings = []
        for doc in tqdm.tqdm(split_data, desc="Embedding documents"):
            print(doc.page_content)
            embedding = embedding_model.embed_documents([doc.page_content])
            embeddings.append(embedding[0])
        
        embeddings = np.array(embeddings)
        embeddings_str = [','.join(map(str, embedding)) for embedding in embeddings]
        with open('embeddings.txt', 'w') as f:
            for embedding_str in embeddings_str:
                f.write(embedding_str + '\n')
        print("Finished embedding")
        return embeddings, embeddings.shape
    
    def create_vector(self, embedding, shape):
        print("Creating vector store...")
        batch_size = 128

        ids = [str(i) for i in range(shape[0])]

        # create list of (id, vector, metadata) tuples to be upserted
        to_upsert = list(zip(ids, embedding))

        for i in range(0, shape[0], batch_size):
            i_end = min(i+batch_size, shape[0])
            self.index.upsert(vectors=to_upsert[i:i_end])

        print("Finished creating vector store")
        # let's view the index statistics
        print(self.index.describe_index_stats())



    # Returns relevant, trimmed, and prompted input for model via vector similarity search
    def retrieve_context(self, query):
        role_prompt = f"You are an expert on Yelp recommendataion system. You will be asked a question about restaurant (most likely asking for restaurant recommendations) and you will respond with a relevant answer. We will provide you with some reviews with different customers to assist your decision making process."

        query = self.vector_store.similarity_search(query, self.top_k)

        retrieval = "\n".join(doc.page_content for doc in query)

        return f"{role_prompt}\nThe following are a few reviews related to the user's query:{retrieval}\nThis following is the user query: {query}"



    # interact with the LLM and update conversation history
    def run_chat(self, user_input):
        enhanced_input = self.retrieve_context(user_input)
        print(enhanced_input)
        self.update_history("user", enhanced_input)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            stream=True
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response