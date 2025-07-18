# Graph-RAG for Scientifixc Q&A

This project implements a hybrid graph + retrieval augmented generation for answering question over scientific papers on arXiv.

## Steps
- Extract papers from arXiv
- FAISS-based semantic retrieval
- Graph traversal to enrich prompt
- LLM generation using Open (working on adding Qwen as well)
- Evaluation with BERT

## How to run
```
mkdir results/ 
python pipeline.py
```

Some requirements are added in `requirements.txt`. The list is not exclusive.
