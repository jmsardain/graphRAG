# Graph-RAG for Scientific Q&A

This project implements a **hybrid graph + retrieval augmented generation (RAG)** system for answering questions over scientific papers on arXiv. It now features a **knowledge graph enhanced with Graph Neural Networks (GNNs)** for deeper reasoning and more robust retrieval.

## Features
- Extracts papers from arXiv
- Builds a knowledge graph of scientific concepts, papers, and relationships
- Uses GNNs to learn node/entity representations from the graph structure
- FAISS-based semantic retrieval of relevant text chunks
- Graph traversal and GNN-based reasoning to enrich LLM prompts
- LLM generation using OpenAI GPT-4 (Qwen support in progress)
- Evaluation with BERTScore

## Pipeline Steps
1. Extract papers from arXiv
2. Build a knowledge graph (entities, relations, papers)
3. Compute node features (text embeddings) and train a GNN on the graph
4. Use GNN embeddings to identify key entities for each query
5. Retrieve relevant text chunks with FAISS
6. Construct a prompt for the LLM using both retrieved text and knowledge graph context (including GNN-derived entities)
7. Generate an answer with the LLM
8. Evaluate answer quality with BERTScore

## How to Run
```bash
mkdir -p results/
python pipeline.py
```

### Or Run with Docker (Recommended for Reproducibility)

1. Build the Docker image:
   ```bash
   docker build -t graphrag .
   ```
2. Run the container (replace YOUR_OPENAI_API_KEY with your actual key):
   ```bash
   docker run -e OPENAI_API_KEY=YOUR_OPENAI_API_KEY graphrag
   ```

This will run the pipeline in a fully reproducible environment, with all dependencies pre-installed.

## Requirements
See `requirements.txt` for dependencies. Notably, you will need:
- torch, torch-geometric, dgl (for GNNs)
- sentence-transformers, faiss-cpu, openai, bert-score, networkx, spacy

## Visualizing the Knowledge Graph and GNN Embeddings

You can visualize the knowledge graph structure and (optionally) the learned GNN embeddings. Here are some example snippets:

### Visualize the Knowledge Graph (NetworkX + PyVis)
```python
import networkx as nx
from pyvis.network import Network
from graph_builder import build_knowledge_graph, fetch_arxiv_papers

papers = fetch_arxiv_papers("graph neural networks", max_results=5)
G = build_knowledge_graph(papers)
net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")
```

### Visualize GNN Embeddings (2D projection with matplotlib)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gnn_module import train_gnn_and_get_embeddings

embeddings = train_gnn_and_get_embeddings(G)
X = np.stack(list(embeddings.values()))
labels = list(embeddings.keys())
X_2d = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1])
for i, label in enumerate(labels):
    plt.annotate(label, (X_2d[i,0], X_2d[i,1]), fontsize=8)
plt.title("GNN Node Embeddings (PCA projection)")
plt.show()
```

---

Feel free to extend or modify the pipeline (in a new branch of course 😊 ) !
