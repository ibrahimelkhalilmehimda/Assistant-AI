from main import agentic_rag
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
class Query(BaseModel):
      question: str


@app.post("/api/chat")
async def chat_endpoint(playload: Query):
          init_state={
                "source":'',
                "query":playload.question,
                "context":'',
                "response":'',
                "is_relevant":'',
                "prompt":'',
                "iteration":0 }
          final_state = agentic_rag.invoke(init_state)

          return  {"res":final_state["response"]}
          
if __name__ == "__main__":
    
    uvicorn.run("api:app", host="localhost", port=8000)
