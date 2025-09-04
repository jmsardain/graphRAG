from typing import Dict, List, Tuple, Optional, Any, Callable
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from graph_builder import fetch_arxiv_papers, getListOfReferences, createGraph, graph_traversal
from utils import create_prompt
from models import OpenAIClient


def main(query, max_results, draw):

    paper_ids = fetch_arxiv_papers(query=query, max_results=max_results)

    dict_papers = {}

    for paper in paper_ids:
        dict_papers[paper] = getListOfReferences(paper)
        # print(paper)
        # print(dict_papers[paper])

    graph = createGraph(dict_papers)
    if draw:
        nx.draw(graph, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
        plt.savefig('plot.png')
        return
    ## Use vector approach
    #########################
    ## create embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paper_embeddings = []
    all_papers = list(graph.nodes)

    for paper in all_papers:
        abstract = graph.nodes[paper]['abstract']
        embedding = model.encode(abstract, convert_to_numpy=True)
        paper_embeddings.append(embedding)


    paper_embeddings = np.array(paper_embeddings)
    dim = paper_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(paper_embeddings)

    # use faiss to find most similar papers (semantically)
    query_embedding = model.encode(query, convert_to_numpy=True)
    _, indices = index.search(np.array([query_embedding]), k=5) ## use only top-k = 5 .. this can be changed
    relevant_vector_papers = [all_papers[i] for i in indices[0]]


    ### Use graph approach (traverse the graph)
    ####################################################
    # Graph-based Retrieval: Retrieve relevant papers using BFS
    relevant_graph_papers = graph_traversal(dict_papers, graph=graph)
    # print(f"Paper from faiss: {relevant_vector_papers}")
    # print(f"Paper from graph: {relevant_graph_papers}")

    ## metric to just compare number of papers
    ## precision: how many relevant papers retrieved are in the all papers / length of all papers
    precision_vector = len(set(relevant_vector_papers) & set(all_papers)) / len(all_papers)
    precision_graph  = len(set(relevant_graph_papers) & set(all_papers)) / len(all_papers)

    # Generate LLM Prompts for both approaches
    graph_prompt = create_prompt(query, relevant_graph_papers, graph)
    vector_prompt = create_prompt(query, relevant_vector_papers, graph)

    # print("Graph-based prompt:\n", graph_prompt)
    # print("\nVector-based prompt:\n", vector_prompt)

    client = OpenAIClient(model="gpt-4o-mini", temperature=0)
    summary_graph  = client.generate(graph_prompt)
    summary_vector = client.generate(vector_prompt)

    # print(f"Summary graph: {summary_graph}")
    # print(f"Summary vector: {summary_vector}")

    with open("results.txt", "w") as f:
        f.write(f"## Vector-based approach:\n")
        f.write(f"Precision: {precision_vector} ## relevant papers / all papers \n")
        f.write(f"Summary: {summary_vector}\n")
        f.write("\n")
        f.write(f"## Graph-based approach:\n")
        f.write(f"Precision: {precision_graph} ## relevant papers / all papers \n")
        f.write(f"Summary: {summary_graph}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="graph neural networks jet tagging particle physics", help="arXiv search query.")
    parser.add_argument("--max", type=int, default=2, help="Maximum amount of arXiv papers from which to extract references.")
    parser.add_argument("--draw", action='store_true', default=False, help="Draw graph from papers.")
    args = parser.parse_args()

    main(args.query, args.max, args.draw)
    pass
