
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# 1. Initialize Phi-3 Mini
llm = ChatOllama(
    model="phi3:mini",
    temperature=0.2
)

# 2. Define the agent state
class AgentState(TypedDict):
    question: str
    answer: str

# 3. Define the agent node
def generate_answer(state: AgentState):
    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }

# 4. Create the LangGraph workflow
graph_builder = StateGraph(AgentState)

graph_builder.add_node("generate", generate_answer)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

# 5. Compile the graph
agent = graph_builder.compile()

# 6. Run the agent
result = agent.invoke({
    "question": "Explain how neural networks work.",
    "answer": ""
})

print("\nAI Agent Response:\n")
print(result["answer"])
