import arxiv
import spacy
import networkx as nx
from utils import extract_entities_relations

def fetch_arxiv_papers(query="graph neural networks", max_results=5):
    search = arxiv.Search(query=query, max_results=max_results)
    return [(r.title, r.summary, r.entry_id, [a.name for a in r.authors], r.published) for r in search.results()]

def build_knowledge_graph(papers):
    nlp = spacy.load("en_core_web_sm")  # use en_core_sci_lg for better results ??
    G = nx.DiGraph()
    for title, abstract, url, authors, date in papers:
        text = title + ". " + abstract
        entities, relations = extract_entities_relations(text, nlp)
        for ent in entities:
            G.add_node(ent, type="entity")
        for head, rel, tail in relations:
            G.add_edge(head, tail, relation=rel)
        G.add_node(title, type="paper", metadata={"url": url, "authors": authors, "date": str(date)})
    return G
