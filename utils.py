

def create_prompt(query, relevant_papers, graph):

    context = ""

    for paper_id in relevant_papers:
        authors = ', '.join(graph.nodes[paper_id]['authors'])
        abstract = graph.nodes[paper_id]['abstract']
        context += f"Paper id: {paper_id}\nAuthors: {authors}\nAbstract: {abstract}\n\n"

    # prompt = f"Query: {query}\n\nContext:\n{context}Answer the query based on the context above."
    prompt = f""" You are a research assistant helping with scientific literature analysis.
    The user will ask you a research-related question. You have access to relevant context, which includes a list of scientific papers.
    Each paper in the context contains the following information:
        - Paper id, is the arxiv identifier of the paper
        - Authors, is the list of authors of the paper
        - Abstract, is the abstract of the paper.

    Use arXiv's search functionality to retrieve relevant data.

    Question:
    {query}

    Context:
    {context}

    Use the context provided to answer the question in this following . Be sure to:
        - cite the papers referenced in the context
        - add information retrieved from arXiv to ensure accuracy
        - Provide detailed and well supported answers based on the papers.

    Provide a concise summary in one small paragraph.
    """
    return prompt
