import faiss
from sentence_transformers import SentenceTransformer
from utils import chunk_text, build_prompt
from openai import OpenAI
import os
import numpy as np


def embed_chunks(papers):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk for _, abstract, *_ in papers for chunk in chunk_text(abstract)]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return texts, embeddings, index

def hybrid_query(query, graph, texts, index, embeddings, top_k=5, gnn_context=None):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_embed = model.encode([query], convert_to_tensor=True)[0].cpu().numpy()
    D, I = index.search(q_embed.reshape(1, -1), top_k)
    retrieved_chunks = [texts[i] for i in I[0]]

    # Graph traversal
    connected = set()
    for node in graph.nodes:
        if query.lower() in node.lower():
            connected.update(graph.neighbors(node))
    entity_context = ", ".join(connected)

    prompt = build_prompt(query, retrieved_chunks, entity_context, gnn_context=gnn_context)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Placeholder for GNN-based retrieval integration
def retrieve_with_gnn(query, graph, gnn_node_embeddings, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(query, convert_to_numpy=True)
    # Compute cosine similarity between query and each node embedding
    similarities = {}
    for node, emb in gnn_node_embeddings.items():
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
        similarities[node] = sim
    # Get top_k nodes/entities
    top_nodes = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
    return top_nodes
