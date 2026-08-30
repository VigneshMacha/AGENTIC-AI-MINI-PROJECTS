import os
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END

load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.1)


def merge_score_dicts(existing:dict,newUpdate:dict)->dict:
    if existing is None:
        return newUpdate
    return {**existing,**newUpdate}


class AnalyzerState(TypedDict):
    raw_text:str
    safety_scores:Annotated[dict[str,int],merge_score_dicts]

def toxicity_node(state: AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}

def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}


graph=StateGraph(AnalyzerState)

graph.add_node("toxicity_node",toxicity_node)
graph.add_node("copyright_node",copyright_node)
graph.add_node("culture_node",culture_node)

graph.add_edge(START,"toxicity_node")
graph.add_edge(START,"copyright_node")
graph.add_edge(START,"culture_node")

graph.add_edge("toxicity_node",END)
graph.add_edge("copyright_node",END)
graph.add_edge("culture_node",END)


app=graph.compile()

text=input("\n\nEnter the raw_text:")
initial_state={
    "raw_text":text,
    "safety_scores":{}
}


response=app.invoke(initial_state)

print("Safety Scores are:\n",response["safety_scores"])