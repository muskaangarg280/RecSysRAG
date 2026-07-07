# Project Title:
Agentic RecSysRAG: A Citation-Grounded Research Assistant for Recommender Systems

Goal:
Build a RAG-based research assistant that answers technical questions about recommender systems using a fixed corpus of research papers. The system should retrieve relevant paper chunks, generate grounded answers, and cite sources. After the baseline RAG system works, an agentic layer will be added using LangGraph to critique retrieval quality, rewrite queries when needed, and route between local corpus search and optional live arXiv search.

Initial Scope:
The corpus will focus on recommender-systems literature, including collaborative filtering, matrix factorization, implicit feedback, neural collaborative filtering, two-tower models, Wide & Deep learning, candidate generation, ranking, cold-start, and sequential recommendation models such as SASRec and BERT4Rec.

Baseline System:
The baseline system will use PDF extraction, text chunking, embeddings, FAISS vector search, and a hosted LLM API to answer questions with citations.

Agentic Extension:
The agentic system will add a retrieve-critique-retry loop. It will judge whether the retrieved chunks are sufficient. If they are weak, the system will rewrite the query and retrieve again before generating the final answer.

Evaluation:
The system will be evaluated using a fixed set of 15 technical questions. Evaluation will compare baseline RAG and agentic RAG using retrieval relevance, citation accuracy, answer faithfulness, latency, and number of retrieval attempts.
