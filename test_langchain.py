
from langchain_ollama import ChatOllama

# Initialize the local Phi-3 Mini model
llm = ChatOllama(
    model="phi3:mini",
    temperature=0.2
)

# Send a question to the model
response = llm.invoke(
    "Explain the difference between Artificial Intelligence and Machine Learning."
)

# Display the response
print("\nPhi-3 Mini Response:\n")
print(response.content)
