from openai import OpenAI
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
import tiktoken

# load env
load_dotenv()


TIP_FILE = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_tips_dense.csv"
REVIEW_FILE = "C:/Users/zhuan/Desktop/Cornell Tech/School Work/Yelp RAG/filtered_reviews_dense.csv"

class YelpExpert():

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)

        # Constants
        self.model = "gpt-3.5-turbo-1106"
        self.max_tokens = 16000
        self.trim_token_limit = self.max_tokens // 3
        self.top_k = 10

        self.tip_summary = self.get_summary(TIP_FILE)
        self.review_summary = self.get_summary(REVIEW_FILE)
        

        self.vector_store = self.create_vector(self.tip_summary, self.review_summary)

        self.conversation_history = []
        self.conversation_tokens = 0
        self.encoding = tiktoken.encoding_for_model(self.model)

    def get_summary(self, file_path):
        output = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # print(row)
                output.append(row[0])
        return output

    def create_vector(self, tip_summary, review_summary):
        '''create vector store where each entry is a line of review or tip'''
        print("Creating vector...")
        split_data = [Document(page_content=x) for x in tip_summary]
        print("length of split data: ",len(split_data), " length of each chunk: ", len(split_data[0].page_content))
        # print(split_data[0].page_content)
        split_data = split_data[:100]
        # os.system('pause')

        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_data, embedding=embeddings)
        print("Vector complete!")
        index = vectorstore.index
        stored_vectors = index.reconstruct_n(0, index.ntotal)
        # print("Stored vectors:", stored_vectors)

        # Save the stored vectors to a text file
        np.savetxt('stored_vectors.txt', stored_vectors)
        return vectorstore

    # Returns relevant, trimmed, and prompted input for model via vector similarity search
    def retrieve_context(self, query):
        role_prompt = f"You are an expert on Yelp recommendataion system. You will be asked a question about restaurant (most likely asking for restaurant recommendations) and you will respond with a relevant answer. We will provide you with some reviews with different customers to assist your decision making process."

        # embeddings = OpenAIEmbeddings()
        # query_vector = embeddings.embed_query(query)

        query = self.vector_store.similarity_search(query, self.top_k)


        retrieval = "\n".join(doc.page_content for doc in query)
        # readme_response = self.trim(readme_string)
        # file_response = self.trim(file_string)

        # readme_response = self.trim(self.readme_vector.similarity_search(query)[0].page_content)
        # file_response = self.trim(self.file_vector.similarity_search(query)[0].page_content)

        # print(f"{role_prompt}\nThe following are a few reviews related to the user's query:{retrieval}\nThis following is the user query: {query}")

        return f"{role_prompt}\nThe following are a few reviews related to the user's query:{retrieval}\nThis following is the user query: {query}"

    # Trim text by number of tokens to obey context window size
    def trim(self, text):
        tokens = self.encoding.encode(text)
        if len(tokens) > self.trim_token_limit:
            trimmed_tokens = tokens[:self.trim_token_limit]
            text = self.encoding.decode(trimmed_tokens)
        return text


    def token_count(self, text):
        return len(self.encoding.encode(text))


    # add conversation to history and keep history size below maxtokens
    def update_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        self.conversation_tokens += self.token_count(content)

        while self.conversation_tokens > self.max_tokens and self.conversation_history:
            removed_entry = self.conversation_history.pop(0)
            self.conversation_tokens -= self.token_count(removed_entry['content'])


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

        self.update_history("assistant", response)
        return response