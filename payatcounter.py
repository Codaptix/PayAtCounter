import pandas as pd
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(all_splits, embeddings)

# ─────────────────────────────────────────────
# Step 3: Define State
# ─────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    user_response: str

llm = OllamaLLM(model="mistral")

# ─────────────────────────────────────────────
# Step 4: Nodes
# ─────────────────────────────────────────────
def correct_grammar(state: State) -> State:
    prompt = (
        f"Correct the grammar of the following sentence. Respond with ONLY the corrected sentence:\n\n{state['question']}"
    )
    corrected = llm.invoke(prompt).strip()
    return {"question": corrected}

def retrieve(state: State) -> State:
    docs = vector_store.similarity_search(state["question"], k=4)
    return {"question": state["question"], "context": docs}

def generate(state: State) -> State:
    context_text = "\n\n".join([doc.page_content for doc in state["context"]])
    prompt = (
        f"You are a concise store assistant. Based on the info below, answer the customer’s question in one or two sentences.\n\n"
        f"Inventory Info:\n{context_text}\n\n"
        f"Customer Question: {state['question']}\n"
        f"Answer only based on what's in the inventory.\n"
        f"Final Answer:"
    )
    answer = llm.invoke(prompt)
    follow_up = " Would you like to buy it? (Yes/No)"
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": answer + follow_up
    }

def get_user_response(state: State) -> State:
    print("🤖", state["answer"])
    response = input("🧑 Your response: ")
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": state["answer"],
        "user_response": response.lower().strip()
    }

def coupon_agent(state: State) -> State:
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": "🎉 Coupon applied successfully! Proceed to counter or checkout.",
        "user_response": state["user_response"]
    }

def end_node(state: State) -> State:
    final_answer = state["answer"]
    if "Would you like to buy it?" in final_answer:
        final_answer = final_answer.split("Would you like to buy it?")[0].strip()
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": final_answer,
        "user_response": state["user_response"]
    }

def route_response(state: State) -> str:
    return "coupon_agent" if state["user_response"] == "yes" else "end"

# ─────────────────────────────────────────────
# Step 5: Build LangGraph
# ─────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("grammar_corrector", correct_grammar)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("get_user_response", get_user_response)
graph_builder.add_node("coupon_agent", coupon_agent)
graph_builder.add_node("end", end_node)

graph_builder.set_entry_point("grammar_corrector")
graph_builder.add_edge("grammar_corrector", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "get_user_response")
graph_builder.add_conditional_edges("get_user_response", route_response)
graph_builder.add_edge("coupon_agent", "end")

graph = graph_builder.compile()

# ─────────────────────────────────────────────
# Run Chat Loop
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Chatbot ready! Ask me about a product (type 'exit' to quit)\n")
    while True:
        q = input("💬 Question: ")
        if q.lower().strip() in ("exit", "quit"):
            break
        result = graph.invoke({"question": q})
        print("✅", result["answer"], "\n")
