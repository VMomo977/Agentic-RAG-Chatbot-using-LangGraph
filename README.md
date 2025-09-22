---
title: "Agentic RAG Sport Advisor Chatbot}"
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
app_file: src/chatbot_nodes.py
pinned: false
license: cc-by-4.0
---

---

# Agentic RAG Sport Advisor Chatbot

This project is a prototype of an **Agentic RAG (Retrieval-Augmented Generation) chatbot**. The application demonstrates core agentic principles by autonomously deciding which tools to use to answer user queries, integrating a RAG pipeline to access document knowledge sources, and utilizing a public knowledge base via Google Search.

---

## Core Concepts

### Agentic Behavior
The chatbot acts as an autonomous agent. Instead of simply generating a response, it first analyzes the user's intent and decides on the most appropriate action. This includes choosing from a set of specialized tools or a general conversation flow.

### Retrieval-Augmented Generation (RAG)
For questions requiring specific, factual information, the agent can retrieve relevant documents from a local vector database and use that information to formulate a grounded response.

### Tool Use
The system is equipped with several tools that allow it to perform specific, pre-defined tasks, such as looking up structured data or performing an external search.

---

## Architecture

The application is built using the **LangGraph framework**, which provides a structured way to define the agent's behavior as a state machine. The graph's state is maintained by a `SportAdvicerState` object that stores the conversation history.

**Key components of the LangGraph pipeline:**

- **chatbot Node:**  
  The entry point and main logic node. It receives a user's query, determines the user's intent (equipment, skills, document, sports_list, or general), and routes the request to the appropriate tool or invokes the LLM directly for general questions.

- **tools Node:**  
  Executes the specific tool chosen by the chatbot node.

- **Conditional Edges:**  
  Routes the flow from the chatbot node to either the tools node (if a specific tool is needed) or terminates the graph (`__end__`) after the response has been generated.

---

## System Components

### RAG Pipeline (`rag_pipeline.py`)
Responsible for setting up the document retrieval system:

- Loads PDF documents from `rag_docs/` using `DirectoryLoader`.
- Splits documents into manageable chunks via `RecursiveCharacterTextSplitter`.
- Uses **ChromaDB** as the local vector store for document chunks and embeddings.
- Generates vector representations with `CohereEmbeddings`.
- Checks for an existing Chroma database to avoid re-processing documents on every run.

### Chatbot Logic and Tools (`chatbot_nodes.py`)
Defines the agent's behavior and tools:

- **LLM:** Uses `gemini-2.5-flash-lite`, a lightweight free-to-use model from Google's Gemini family.

**Intent and Parameter Extraction:**
- `detect_intent(query)`: Classifies user queries into a specific category.
- `extract_sport_name(query)`: Extracts the relevant sport name for tools.

**Tools:**
- `get_sports()`: Returns a list of sports from a local CSV.
- `get_skills_by_sport(sport: str)`: Retrieves the top 3 highest-rated skills for a sport.
- `get_document_answer(query: str)`: Core RAG tool using a retriever to find relevant document chunks and augment LLM prompts.
- `get_equipment_by_sport(sport: str)`: Performs a Google Search to find sport equipment information.

---

## How to Run the Application

### Prerequisites
- Python 3.9+
- pip
- Google API Key
- Cohere API Key

### Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Place your PDF documents in the rag_docs/ directory.
4. Create a .env file in the root directory and add your Google API key and Cohere API Key:

```bash
GOOGLE_API_KEY="your_api_key_here"
COHERE_API_KEY="your_api_key_here"
```
### Execution
Run the main application:
```bash
python chatbot_nodes.py
```
The application will start a Gradio web interface, accessible via your browser to interact with the chatbot.

---

## Evaluation and Future Work

### Bottlenecks and Performance
- Multiple LLM calls for intent detection and sport name extraction may introduce latency.
- RAG pipeline performance depends on document quality and chunk_size / chunk_overlap parameters.

### Suggested Improvements
- Advanced Tool Calling: Use a single LLM call to decide which tool to call with what parameters.
- Performance Benchmarking: Test queries to measure end-to-end latency and response accuracy.
- New Tools: Extend capabilities with weather updates, sports news, or game schedules.

---

## Example Human Questions to Tools Mapping
This section provides examples of user questions that would trigger the various tools within the chatbot.

- `get_sports()`: "Recommend me sports."

- `get_skills_by_sport(sport: str)`: "What skills are needed for football?"

- `get_document_answer(query: str)`:

  - "How can I be successful in football based on documentations?"
  - "How can influence a football match the location based on the documentation?"
  - "What factors influence the judging in gymnastics based on the documentations?"
  - "What are some specific deductions a gymnast might receive during a competition based on the documentations?"

- `get_equipment_by_sport(sport: str)`: "What gears are needed for football?"

---

## References and Citations

**RAG Input References:**
- **Dataset license**: Creative Commons Attribution 4.0 International Public License (CC-BY 4.0)
- `rag_docs/TOSSJ-11-3.pdf`:
  
Lepschy H, Wäsche H, Woll A. How to be Successful in Football: A Systematic Review . Open Sports Sci J, 2018; 11: . http://dx.doi.org/10.2174/1875399X01811010003

- `rag_docs/TOSSJ-12-1.pdf`:

Mack M, Bryan M, Heyer G, Heinen T. Modeling Judges’ Scores in Artistic Gymnastics . Open Sports Sci J, 2019; 12: . http://dx.doi.org/10.2174/1875399X01912010001

**Sport tool input csv file references:**
- **Dataset license**: CC0: Public Domain
- `sport_tool_docs/toughestsport.csv`:
  
[Ranking sports by skill requirement](https://www.kaggle.com/datasets/jainaru/ranking-sports-by-skill-requirement)
