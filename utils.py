def extract_entities_relations(text, nlp):
    doc = nlp(text)
    entities = set(ent.text for ent in doc.ents)
    # Dummy relation extraction: could be improved
    relations = [(ent1.text, "related_to", ent2.text) for ent1 in doc.ents for ent2 in doc.ents if ent1 != ent2]
    return list(entities), relations

def chunk_text(text, max_tokens=150):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def build_prompt(query, chunks, graph_context):
    context_bullets = ''.join(['- ' + c + '\n' for c in chunks])
    return f"""
Answer the following scientific question based on retrieved information.

Question: {query}

Relevant Context:
{context_bullets}

Related Entities:
{graph_context}

Be accurate and cite evidence from the context.
"""
