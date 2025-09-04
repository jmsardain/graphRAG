import arxiv
import requests
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any, Callable
from pdfminer.high_level import extract_text
import re
import networkx as nx


def fetch_arxiv_papers(query="graph neural networks", max_results=1):
    search = arxiv.Search(query=query, max_results=max_results)
    # return [(r.title, r.summary, r.entry_id, [a.name for a in r.authors], r.published) for r in search.results()]
    paper_ids = [r.entry_id for r in search.results()]
    paper_ids_only = [re.sub(r'^http://arxiv.org/abs/', '', paper_id) for paper_id in paper_ids]
    return paper_ids_only

def getListOfReferences(paper_id: str) -> List[str]:
    """
    Tool to get the references from pdf. Converts pdf to text and extract arxiv references.
    This could be improved by using APIs such as connectedpapers and using doi.
    """
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    # Download PDF
    response = requests.get(pdf_url)
    pdf_data = BytesIO(response.content)

    # Extract text
    text = extract_text(pdf_data)

    arxiv_ids = re.findall(r'arXiv:\s?\d{4}\.\d{4,5}(?:v\d+)?', text)
    arxiv_ids_only = [re.sub(r'^arXiv:\s*', '', arxiv_id) for arxiv_id in arxiv_ids]
    more_ids = re.findall(r'\[\d+\.\d+\]', text) ## sometimes arxiv references are just put in []. no arxiv: before them.
    more_ids_only = [re.sub(r'[\[\]]', '', more_id) for more_id in more_ids]
    arxiv_ids_only.extend(more_ids_only)
    return arxiv_ids_only

def getMetaData(arxiv_id):
    search = arxiv.Search(id_list=[arxiv_id])
    for result in search.results():
        authors = [author.name for author in result.authors]
        abstract = result.summary
        return authors, abstract
    return [], ""  # Return empty if not found

def createGraph(dict_papers):
    graph = nx.Graph()

    for paper in dict_papers:
        authors, abstract = getMetaData(paper)
        graph.add_node(paper, authors=authors, abstract=abstract)

    for paper, refs in dict_papers.items():
        for ref in refs:
            if ref not in graph.nodes:
                authors, abstract = getMetaData(ref)
                graph.add_node(ref, authors=authors, abstract=abstract)
            graph.add_edge(paper,ref)

    return graph



def graph_traversal(dict_papers, graph, max_hops=2):

    relevant_papers = set()
    for paper in dict_papers:
        queue = [paper]
        hops  = 0
        while queue and hops<max_hops:
            next_queue = []

            for paper in queue:
                relevant_papers.add(paper)
                next_queue.extend(graph.neighbors(paper))
            queue = next_queue
            hops +=1

    return list(relevant_papers)
