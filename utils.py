def extract_entities_relations(text, nlp):
    doc = nlp(text)
    entities = set(ent.text for ent in doc.ents)
    # Dummy relation extraction: could be improved
    relations = [(ent1.text, "related_to", ent2.text) for ent1 in doc.ents for ent2 in doc.ents if ent1 != ent2]
    return list(entities), relations

def chunk_text(text, max_tokens=150):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def build_prompt(query, chunks, graph_context, gnn_context=None):
    context_bullets = ''.join(['- ' + c + '\n' for c in chunks])
    gnn_section = f"\nKey Entities from the Knowledge Graph (via GNN reasoning):\n{gnn_context}\n" if gnn_context else ""
    return f"""
You are an expert scientific assistant. Use both retrieved text and structured knowledge from a knowledge graph to answer the following question. The knowledge graph encodes relationships between scientific concepts, papers, and entities, and has been processed with graph neural networks (GNNs) to identify key relevant entities.

Question: {query}

Relevant Context:\n{context_bullets}

Related Entities from Graph Traversal:\n{graph_context}\n{gnn_section}Be accurate, cite evidence from the context, and leverage both the retrieved text and the knowledge graph for your answer.
"""
