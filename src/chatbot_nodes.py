
import os
from pathlib import Path
import traceback
from typing import Annotated, List, Union
from typing_extensions import TypedDict

import pandas as pd
import gradio as gr

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from rag_pipeline import load_or_create_vector_store

from google import genai
from google.genai import types
from google.api_core import retry

from dotenv import load_dotenv

# ---------------------------
# --- Setup Google API ---
# ---------------------------
# Modify this load_dotenv in the future
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Retry policy
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)

# ---------------------------
# --- Config ---
# ---------------------------
model_name = "gemini-2.5-flash-lite" #"gemini-2.0-flash-lite"#"gemini-2.0-flash"
base_dir = Path(__file__).resolve().parent.parent
doc2_path = str( base_dir / "sport_tool_docs/toughestsport.csv")
search_kwargs_k = 5
search_kwargs_fetch_k = 10

# ---------------------------
# --- RAG Setup ---
# ---------------------------
vector_store = load_or_create_vector_store()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": search_kwargs_k,
        "fetch_k": search_kwargs_fetch_k
        }
)

# ---------------------------
# --- Sports Data Setup ---
# ---------------------------
sports_by_skills = pd.read_csv(doc2_path)
sports_by_skills.columns = sports_by_skills.columns.str.lower()
sports_by_skills['sport'] = sports_by_skills['sport'].str.lower()

# ---------------------------
# --- LangGraph Tools ---
# ---------------------------
@tool
def get_sports() -> str:
    """Return a structured list of sports from the dataset."""
    
    prompt = """Parse the provided sports into a structured list where each line has indentation and starts with a category, followed by a colon, 
    and then a comma-separated list of sports within that category. 
    If a sport has no obvious category, group it under "General"
    
    EXAMPLE:
    Provide me sport options
    
    Answer: 
    - Ball games: 
        - Football, Baskettball
    - Skiing: 
        - Alpine, Nordic
    - General: 
        - Boxing, Water polo
    
    """
    sports = sports_by_skills["sport"].tolist()
    response = client.models.generate_content(model=model_name, contents=[prompt, sports])
    return response.candidates[0].content.parts[0].text.strip()

@tool
def get_document_answer(query: str) -> str:
    """Retrieve an answer from documents with a grounded paraphrase."""
    
    try:
        results = retriever.invoke(query)
        
        if not results:
            return "I could not find any relevant information in the documents."

        # Combine the retrieved chunks for context
        combined_text = "\n---\n".join([r.page_content for r in results])

        prompt = f"""
        Answer the question based on the following documents. 
        If the information is not available, state that you cannot find the answer in the provided documents.
        
        Chunks:
        {combined_text}
        
        Question: {query}
        Answer:
        """

        # Call the LLM
        response = client.models.generate_content(
            model=model_name,  
            contents=[prompt]
        )

        if response.candidates:
            answer_text = response.candidates[0].content.parts[0].text.strip()
            return answer_text
        else:
            return "I could not generate an answer."

    except Exception as e:
        return f"RAG error: {e}"

@tool
def get_skills_by_sport(sport: str) -> str:
    """Get the sport name. Return: The top 3 highest skill rates."""
    sport = sport.lower().strip()
    skill_rates = sports_by_skills.loc[sports_by_skills['sport'] == sport]

    if skill_rates.empty:
        return f"No data found for sport '{sport}'. Please check the spelling or try another sport."

    skills_only = skill_rates.drop(columns=['sport', 'total', 'rank'])
    transposed = skills_only.T
    col = transposed.columns[0]
    top_3_skills = transposed.nlargest(3, col)
    top_3_skill_names = "\n".join(f"{skill}" for skill, value in top_3_skills[col].items())
    
    return f"Top 3 skills for {sport.capitalize()}:\n{top_3_skill_names}"

@tool
def get_equipment_by_sport(sport: str) -> str:
    """Get the equipment list for a sport using a google search grounded prompt."""
    
    sport = sport.lower()
    prompt = """Parse a customer's sport equipment question to the list:
    EXAMPLE: What are the necessary equipment for boxing?
    Response: 
    - Mandatory: 1 gloves, 3 socks 
    - Recommended: 1 towel 
    - Fun: resistance bands
    """

    config_with_search = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
    )
    
    contents_text = "What are the necessary equipment for this " + sport + "?"
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, contents_text],
        config=config_with_search,
    )
    return response.candidates[0].content.parts[0].text if response.candidates else "No information found."

# ---------------------------
# --- Tool Node ---
# ---------------------------
tools_list = [get_sports, get_document_answer, get_skills_by_sport, get_equipment_by_sport]
tool_node = ToolNode(tools_list)

# ---------------------------
# --- LangGraph LLM ---
# ---------------------------
llm = ChatGoogleGenerativeAI(model=model_name)
llm_with_tools = llm.bind_tools(tools_list, return_direct=True) 

# --- Graph State ---
class SportAdvicerState(TypedDict):
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], list.__add__]

def detect_intent(query: str) -> str:
    """Classify user query into one of: equipment, skills, document, sports_list, general."""
    classification_prompt = f"""
    You are a classifier. 
    Categorize the following user query into exactly ONE of these categories:
    - equipment → if asking about gear, equipment, things needed for a sport
    - skills → if asking about skills, abilities, rankings, requirements for a sport
    - document → if asking about information that may be inside books, PDFs, or retrieved documents
    - sports_list → if asking for a list of sports, categories of sports, or groupings of sports
    - general → if it's a general sports question not fitting the above

    Query: "{query}"

    Answer with one word: equipment, skills, document, sports_list, or general.
    """

    response = client.models.generate_content(
        model=model_name,
        contents=[classification_prompt],
        config=types.GenerateContentConfig(
            temperature=0.0  # deterministic
        )
    )

    if response.candidates:
        return response.candidates[0].content.parts[0].text.strip().lower()
    else:
        return "general"

def extract_sport_name(query: str) -> str:
    """Extract the sport name from a user query."""
    extraction_prompt = f"""
    Extract the single sport name from the following query.
    If multiple sports are mentioned, return the first one.
    If no sport is mentioned, return an empty string.

    Query: "{query}"

    Extracted sport name:
    """
    response = client.models.generate_content(
        model=model_name,
        contents=[extraction_prompt],
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )
    if response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    return ""

def chatbot_node(state: SportAdvicerState) -> SportAdvicerState:
    user_message = state["messages"][-1]
    query = user_message.content
    intent = detect_intent(query)
    sport_name = extract_sport_name(query)

    if intent == "equipment":
        response_text = get_equipment_by_sport.invoke({"sport": sport_name})
        return {"messages": [AIMessage(content=response_text)]}

    elif intent == "skills":
        response_text = get_skills_by_sport.invoke({"sport": sport_name})
        return {"messages": [AIMessage(content=response_text)]}

    elif intent == "document":
        response_text = get_document_answer.invoke({"query": query})
        return {"messages": [AIMessage(content=response_text)]}

    elif intent == "sports_list":
        response_text = get_sports.invoke({})
        return {"messages": [AIMessage(content=response_text)]}

    else:  # general
        messages_with_instruction = [
            HumanMessage(content="""You are a sports advisor chatbot.
            You can answer general sports questions.
            For equipment, skills, document, or sports list queries, tools are used automatically.""")
        ] + state["messages"]

        response = llm_with_tools.invoke(messages_with_instruction)
        return {"messages": [response]}

# Routing: always go to the single tool node if any tool call exists
def should_route_to_tools(state: SportAdvicerState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "__end__"

# ---------------------------
# --- Graph Definition ---
# ---------------------------
graph_builder = StateGraph(SportAdvicerState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", should_route_to_tools)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph_with_rag = graph_builder.compile()

# ---------------------------
# --- Gradio Interface ---
# ---------------------------
def chatbot_interface(message, history):
    langchain_messages = []
    for chat_entry in history:
        if isinstance(chat_entry, list) and len(chat_entry) == 2:
            if chat_entry[0]: langchain_messages.append(HumanMessage(content=chat_entry[0]))
            if chat_entry[1]: langchain_messages.append(AIMessage(content=chat_entry[1]))
        elif isinstance(chat_entry, dict):
            if chat_entry["role"] == "user": langchain_messages.append(HumanMessage(content=chat_entry["content"]))
            elif chat_entry["role"] == "assistant": langchain_messages.append(AIMessage(content=chat_entry["content"]))

    langchain_messages.append(HumanMessage(content=message))
    current_state = {"messages": langchain_messages}
    
    try:
        response_state = graph_with_rag.invoke(current_state)
        bot_response = response_state["messages"][-1].content
        return bot_response
    except Exception as e:
        traceback.print_exc()
        return f"Internal error: {e}"

iface = gr.ChatInterface(
    fn=chatbot_interface,
    chatbot=gr.Chatbot(height=500, type="messages",
                       value=[{"role": "assistant", "content": "Hello! I am your AI Sport Advisor. Ask me anything."}]),
    title="Agentic RAG Sport Advisor Chatbot",
    description="LangGraph chatbot integrated with RAG document retrieval and sports tools.",
    type="messages",
)

if __name__ == "__main__":
    iface.launch(share=True)