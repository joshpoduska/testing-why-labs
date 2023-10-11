import os
import pickle
import random


from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ChatVectorDBChain
from langchain.llms import GPT4All
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain import HuggingFaceHub




def get_chain(vectorstore):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''
    repo_id = "databricks/dolly-v2-3b" 
    llm = HuggingFaceHub(repo_id=repo_id)        
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain



_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about  Domino Data Labs product documentation.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about AI or ML or data science or MLOps or related to Domino Data Lab, politely refrain from answering.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

# Load the embeddings from the pickle file; change the location if needed
with open("faiss_doc_store.pkl", "rb") as f:
    store = pickle.load(f)


chat_history = []

if store:
    qa = get_chain(store)
    user_input = "What are datasources?"
    result = qa({"question": user_input, "chat_history": chat_history})
    answer = result["answer"]
    print(answer)
    chat_history.append((user_input, answer))
        
            
 