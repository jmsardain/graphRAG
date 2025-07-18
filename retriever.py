import faiss
from sentence_transformers import SentenceTransformer
from utils import chunk_text, build_prompt
from openai import OpenAI
import os


def embed_chunks(papers):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk for _, abstract, *_ in papers for chunk in chunk_text(abstract)]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return texts, embeddings, index

def hybrid_query(query, graph, texts, index, embeddings, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # q_embed = model.encode([query])[0]
    q_embed = model.encode([query], convert_to_tensor=True)[0].cpu().numpy()
    D, I = index.search(q_embed.reshape(1, -1), top_k)
    retrieved_chunks = [texts[i] for i in I[0]]

    # Graph traversal
    connected = set()
    for node in graph.nodes:
        if query.lower() in node.lower():
            connected.update(graph.neighbors(node))
    entity_context = ", ".join(connected)

    prompt = build_prompt(query, retrieved_chunks, entity_context)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
