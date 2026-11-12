"importation"
import os
from openai import OpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
# LLM model from Open AI to generate responses
load_dotenv()

api_key = os.getenv("OPEN_AI_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
#api_key = os.getenv("GROQ_API_KEY")
@tool
def llm_response(prompt:str) -> str:
    """" Generate a response using a open ai model """
    from openai import OpenAI
    client_llm = OpenAI(api_key=api_key)
    response = client_llm.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

#victotial DATABASE

reader = PdfReader("pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

chunk_size = 500
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

ids = [f"""doc_{i}""" for i in range(len(chunks))]

client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection(name='medical_qna')

collection.add(
    documents=chunks,  
    ids=ids            
)

# Build the rag pipeline

def retrieve_context_DB(state):
    """Retrieve top documents from ChromaDB Collection."""
    query=state["query"]

    results = collection.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
    state["context"] = context
    return state
def retrieve_context_web_search(state):
    """ Perform web search using the Tavily Search API."""
    query=state["query"]
    tavily_search = TavilySearchResults(topic="general", max_results=1)
    result = tavily_search.invoke({"query": query})
    state["context"] =  result['results'][0]['content']
    return state
    


def router(state):
    """" agentic router to decide which retrieval to use """
    query=state["query"]
    decision_prompt= f"""" 
    You are a reasoning agent. Based on the user's query, you must decide where to retrieve the necessary information. 

    You have three information sources available:
    - pdf search: if about copmliation process 
    - Web Search: Use this for recent information.
    Query: "{query}"

    Choose the most appropriate source to answer the query. Respond with only the name of the chosen option (either "pdf search" or "Web_Search" ), without any extra text, explanation, or punctuation."""
    router_decision= llm_response(decision_prompt)
    state["source"] = router_decision
    return state,

def route_decision(state) -> str:
    return state["source"] 

def check_content_revelance(state):
    """" verify if the retrieved context is relevant to the query """
    query = state["query"]
    context = state["context"]
    relevance_prompt=f"""""
    You are a check context relevance agent. Your task is to determine whether the retrieved context is relevant to the user query or not.

    Retrieved Context:
    "{context}"

   User Query:
   "{query}"

   Evaluation Options:
  - Yes: If the retrieved context is relevant to the user query.
  - No: If the retrieved context is not relevant to the user query.

  Respond ONLY with "Yes" or "No", without any additional text, explanation, or punctuation."""
    relevance_decision_value = llm_response(relevance_prompt)
    state["is_relevent"]=relevance_decision_value
    return state

def build_prompt(state):
   """Construct the RAG-style prompt."""
   query = state["query"]
   context = state["context"]
   prompt = f"""
    Answer the following question using the context below.
    Context:
    {context}
    Question: {query}
    please limit your answer in 50 words.
    """
    
   state["prompt"] = prompt
   print(prompt)
   return state

def call_llm(state):
    """Call your existing LLM function."""
    print("---GENERATE (CALLING LLM)---")
    prompt = state["prompt"]
    answer = llm_response(prompt)
    state["response"] = answer
    return state

def hallucination_test(state):
   """" verifiy if the llm has hallucinted the response"""
   query = state["query"]
   response = state["response"]
   prompt=f"""""
   You are an Answer Verification Agent. Your task is to determine whether the provided LLM response directly answers the user's query accurately and does not contain any hallucinations, fabricated information, or irrelevant details.

User Query:
"{query}"

Generated Response:
"{response}"

Evaluation Options:
- NO: If the response directly answers the query accurately and contains no hallucinations or contradictory information.
- Yes: If the response contains hallucinations, ungrounded information, off-topic details, or fails to answer the question.

Respond ONLY with "Yes" or "No", without any additional text, explanation, or punctuation."""
   state['halucination'] = llm_response(prompt)
   return state

def hallucination(state):
    return state['halucination']

def relevance_itertion(state):
   itr_count = state.get('itrtation',0)
   itr_count+=1
   state["iteration"]= itr_count

   if state["iteration"]>=3:
      print("max possible iteration reached")
      state["is_relevent"]="impossible" 

   return state["is_relevent"]

def handle_impossible(state):
   print("informations not found")
   state["response"]="Sorry, after several attempts, I am unable to find reliable information to answer your question."
   return state

 