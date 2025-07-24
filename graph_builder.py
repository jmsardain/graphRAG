import arxiv
import spacy
import networkx as nx
from utils import extract_entities_relations
from sentence_transformers import SentenceTransformer
import numpy as np

def fetch_arxiv_papers(query="graph neural networks", max_results=5):
    search = arxiv.Search(query=query, max_results=max_results)
    return [(r.title, r.summary, r.entry_id, [a.name for a in r.authors], r.published) for r in search.results()]

def build_knowledge_graph(papers):
    nlp = spacy.load("en_core_web_sm")  # use en_core_sci_lg for better results ??
    G = nx.DiGraph()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for title, abstract, url, authors, date in papers:
        text = title + ". " + abstract
        entities, relations = extract_entities_relations(text, nlp)
        # Node features: entity text embedding
        for ent in entities:
            ent_emb = model.encode(ent, convert_to_numpy=True)
            G.add_node(ent, type="entity", feature=ent_emb)
        for head, rel, tail in relations:
            # Edge features: relation type as categorical index (for now, just 0 for 'related_to')
            G.add_edge(head, tail, relation=rel, feature=np.array([0]))
        # Paper node: use title embedding
        title_emb = model.encode(title, convert_to_numpy=True)
        G.add_node(title, type="paper", metadata={"url": url, "authors": authors, "date": str(date)}, feature=title_emb)
    return G
