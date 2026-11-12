
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import agents as ag

class GraphState(TypedDict):
      source:str
      query:str
      context:str
      response:str
      is_relevant:str
      prompt:str
      iteration:int

graph= StateGraph(GraphState)

graph.add_node("router", ag.router)
graph.add_node("retrieve_context_DB", ag.retrieve_context_DB)
graph.add_node("retrieve_context_web_search", ag.retrieve_context_web_search)
graph.add_node("Relevance_Test", ag.check_context_relevance)
graph.add_node("build_prompt", ag.build_prompt)
graph.add_node("response", ag.call_llm)
graph.add_node("hallucination_test", ag.hallucination_test)
graph.add_node("handle_impossible", ag.handle_impossible)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    ag.route_decision, 
    {
        "pdf search": "retrieve_context_DB",
        "Web_Search": "retrieve_context_web_search",
    }
)
graph.add_edge("retrieve_context_DB", "Relevance_Test")
graph.add_edge("retrieve_context_web_search", "Relevance_Test")
graph.add_conditional_edges(
    "Relevance_Test",
    ag.relevance_itertion,  
    {
        "Yes": "build_prompt",
        "No": "retrieve_context_web_search",
        "impossible":"handle_impossible"
    }
)
graph.add_conditional_edges(
    "response",
    ag.hallucination,  
    {
        "Yes": "retrieve_context_web_search",
        "No": "END",
    }
)
graph.add_edge("build_prompt", "response")
graph.add_edge("handle_impossible", END)

agentic_rag = graph.compile()