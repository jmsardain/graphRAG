from graph_builder import fetch_arxiv_papers, build_knowledge_graph
from retriever import embed_chunks, hybrid_query
from evaluator import evaluate_answers
from gnn_module import train_gnn_and_get_embeddings
from retriever import retrieve_with_gnn

if __name__ == '__main__':
    query = "How do GNNs improve jet tagging in the context of particle physics?"


    ## Get arxiv papers
    papers = fetch_arxiv_papers("graph neural networks jet tagging particle physics", max_results=10)

    ## Build knowledge graph
    G = build_knowledge_graph(papers)

    ## Embeddings and indexing
    texts, embeddings, index = embed_chunks(papers)

    # --- GNN Integration ---
    gnn_node_embeddings = train_gnn_and_get_embeddings(G)
    gnn_results = retrieve_with_gnn(query, G, gnn_node_embeddings, top_k=5)
    print("[GNN Top Entities]:", gnn_results)
    # Use gnn_results as additional context for the prompt
    gnn_context = ", ".join(gnn_results)
    # -----------------------

    ## Run another query (with GNN context)
    # Optionally, you can modify hybrid_query to accept extra context, or just print for now
    answer = hybrid_query(query, G, texts, index, embeddings, top_k=5, gnn_context=gnn_context)
    print("\n[Answer]:", answer)
    print("\n[GNN Context]:", gnn_context)

    ## Evaluate the answers with BERT score
    evaluate_answers([query], [answer])

    # with open("results/answers.txt", "a") as f:
    #     f.write(f"Query: {query}\nAnswer: {answer}")
