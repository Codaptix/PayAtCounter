import pandas as pd
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM


# ─────────────────────────────────────────────────────────────
# Step 1: Load your menu + coupon data
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("smoke_store_menu.csv")  # CSV should have ItemID, ItemName, Category, Price, CouponCode, DiscountType, DiscountValue, MinPurchase

docs: List[Document] = []
for _, row in df.iterrows():
    text = (
        f"Item: {row['ItemName']}, Category: {row['Category']}, Price: ${row['Price']:.2f}, "
        f"Coupon: {row['CouponCode']} gives {row['DiscountValue']}"
        f"{'%' if row['DiscountType']=='Percentage' else '$'} off on minimum purchase of ${row['MinPurchase']:.2f}."
    )
    docs.append(Document(page_content=text, metadata={"id": row["ItemID"]}))

# ─────────────────────────────────────────────────────────────
# Step 2: Split into chunks (if needed)
# ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = splitter.split_documents(docs)

# ─────────────────────────────────────────────────────────────
# Step 3: Generate embeddings and store in Chroma
# ─────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(all_splits, embeddings)

# ─────────────────────────────────────────────────────────────
# Step 4: Define your custom RAG logic
# ─────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Step 4.1: Retrieval step
def retrieve(state: State):
    docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": docs}

# Step 4.2: Generation step using Mistral via Ollama
llm = OllamaLLM(model="mistral")

def generate(state: State):
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])
    prompt = (
        f"You are a concise store assistant. Based on the information below, answer the customer’s question clearly in one or two sentences.\n\n"
        f"Inventory Info:\n{context_text}\n\n"
        f"Customer Question: {state['question']}\n"
        f"Answer only based on what's in the inventory. No assumptions or extra suggestions.\n"
        f"Final Answer:"
    )
    answer = llm.invoke(prompt)
    return {"answer": answer}

# ─────────────────────────────────────────────────────────────
# Step 5: Build LangGraph
# ─────────────────────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

# ─────────────────────────────────────────────────────────────
# Step 6: Run the chatbot REPL
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Chatbot is ready! Ask me about the store (type 'exit' to quit)\n")
    while True:
        q = input("💬 Question: ")
        if q.lower().strip() in ("exit", "quit"):
            break
        response = graph.invoke({"question": q})
        print("✅ Answer:", response["answer"], "\n")
