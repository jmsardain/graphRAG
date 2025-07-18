from graph_builder import fetch_arxiv_papers, build_knowledge_graph
from retriever import embed_chunks, hybrid_query
from evaluator import evaluate_answers

if __name__ == '__main__':
    query = "How do GNNs improve jet tagging in the context of particle physics?"


    ## Get arxiv papers
    papers = fetch_arxiv_papers("graph neural networks jet tagging particle physics", max_results=10)

    ## Build knowledge graph
    G = build_knowledge_graph(papers)

    ## Embeddings and indexing
    texts, embeddings, index = embed_chunks(papers)

    ## Run another query
    answer = hybrid_query(query, G, texts, index, embeddings, top_k=5)
    print("\n[Answer]:", answer)

    ## Evaluate the answers with BERT score
    evaluate_answers([query], [answer])

    # with open("results/answers.txt", "a") as f:
    #     f.write(f"Query: {query}\nAnswer: {answer}")
