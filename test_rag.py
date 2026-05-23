import pytest
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

from ragas.metrics import Faithfulness, AnswerRelevancy

# FIX 2: Import FastAPI TestClient and your app
from fastapi.testclient import TestClient
from main import app 

def test_rag_regression_gate():
    print("\n[CI/CD] Starting RAG Regression Gate...")

    # FIX 3: Spin up the app context to initialize the ML components!
    with TestClient(app):
        # Now we can safely grab the initialized tools from app.state
        query_rewriter = app.state.query_rewriter
        retriever = app.state.retriever
        qa_chain = app.state.qa_chain
        llm = app.state.llm
        embedding_function = app.state.embedding_function
        
        # ==========================================
        # 1. THE GOLDEN DATASET (20 Targeted Questions)
        # ==========================================
        golden_questions = [
            "What problem does FlashAttention-2 primarily address in Transformer models?",
            "What memory complexity does standard attention require?",
            "What is the main architectural innovation introduced in the Transformer paper?",
            "What is the purpose of multi-head attention in Transformers?",
            "What is LoRA in large language model adaptation?",
            "Why is full fine-tuning difficult for very large models like GPT-3?",
            "How does LoRA differ from adapter-based fine-tuning methods?",
            "What is the largest model size described in the Llama 3 paper?",
            "What capabilities are natively supported by Llama 3 models?",
            "What are the two major stages in developing Llama 3?",
            "What optimization technique does Llama 3 use to improve inference efficiency?",
            "What percentage of the final Llama 3 data mix consists of code tokens?",
            "Why does Llama 3 remove markdown markers during preprocessing?",
            "What is TorchDynamo in PyTorch 2?",
            "What does torch.compile provide in PyTorch 2?",
            "What languages does TorchInductor generate code for?",
            "What inference speedup does TorchInductor achieve on NVIDIA A100 GPUs?",
            "Why are eager mode frameworks preferred by many researchers?",
            "What is the main drawback of standard attention implementations?",
            "Why is FlashAttention considered more memory efficient than standard attention?"
        ]

        golden_truths = [
            "FlashAttention-2 addresses the quadratic runtime and memory bottleneck of attention layers in Transformers, especially for long sequence lengths.",
            "Standard attention requires O(N^2) memory complexity because it materializes the attention score and probability matrices.",
            "The Transformer architecture relies entirely on attention mechanisms without using recurrence or convolutions.",
            "Multi-head attention allows the model to attend to information from different representation subspaces simultaneously.",
            "LoRA, or Low-Rank Adaptation, freezes pretrained model weights and injects trainable low-rank matrices into Transformer layers.",
            "Full fine-tuning is difficult because each downstream task requires storing and training a separate massive model with billions of parameters.",
            "LoRA differs from adapter methods by introducing no additional inference latency and directly modifying weight updates through low-rank decomposition.",
            "The largest model described in the Llama 3 paper has 405 billion parameters.",
            "Llama 3 natively supports multilinguality, coding, reasoning, and tool usage.",
            "The two major stages in developing Llama 3 are pre-training and post-training.",
            "Llama 3 uses grouped query attention (GQA) with 8 key-value heads to improve inference efficiency.",
            "Approximately 17% of the final Llama 3 data mix consists of code tokens.",
            "Llama 3 removes markdown markers because experiments showed markdown harmed performance compared to plain text.",
            "TorchDynamo is a Python-level just-in-time compiler introduced in PyTorch 2 for dynamic graph capture.",
            "torch.compile provides graph compilation and compiler optimizations for PyTorch programs while preserving eager execution flexibility.",
            "TorchInductor generates Triton code for GPUs and C++ code for CPUs.",
            "TorchInductor achieves a 2.27× inference geometric mean speedup on NVIDIA A100 GPUs across real-world models.",
            "Researchers prefer eager mode frameworks because they are easier to understand and debug using standard Python tools.",
            "The main drawback of standard attention implementations is the quadratic memory and runtime cost caused by materializing attention matrices.",
            "FlashAttention is more memory efficient because it reduces memory usage from quadratic to linear in sequence length."
         ]

        answers = []
        contexts = []

        # ==========================================
        # 2. GENERATE ANSWERS
        # ==========================================
        print("[CI/CD] Generating answers via Split-Brain pipeline...")
        for q in golden_questions:
            optimized_query = query_rewriter.invoke({"input": q, "chat_history": "No prior conversation."})
            retrieved_docs = retriever.invoke(optimized_query)
            response_text = qa_chain.invoke({"input": q, "context": retrieved_docs})
            
            answers.append(response_text)
            contexts.append([doc.page_content for doc in retrieved_docs])

        data = {
            "question": golden_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": golden_truths
        }
        dataset = Dataset.from_dict(data)

        # ==========================================
        # 3. RUN RAGAS EVALUATION
        # ==========================================
        print("[CI/CD] Grading answers using LLM-as-a-Judge...")
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(strictness=1)],
            llm=llm, 
            embeddings=embedding_function, 
            raise_exceptions=False,
            run_config=RunConfig(max_workers=1, max_retries=3) 
        )
        
        df = result.to_pandas()
        
        # Calculate dataset averages (ignoring NaNs from failed LLM calls)
        avg_faithfulness = df['faithfulness'].mean()
        avg_relevancy = df['answer_relevancy'].mean()
        
        print(f"\n--- CI/CD PIPELINE RESULTS ---")
        print(f"Average Faithfulness: {avg_faithfulness:.4f}")
        print(f"Average Answer Relevancy: {avg_relevancy:.4f}")

        # ==========================================
        # 4. THE REGRESSION GATE (The Magic Step)
        # ==========================================
        THRESHOLD = 0.85
        
        # If the score is below 0.85, this assert will trigger a fatal error and block the pipeline!
        assert avg_faithfulness >= THRESHOLD, f"🚨 REGRESSION DETECTED! Faithfulness dropped to {avg_faithfulness:.4f}"
        assert avg_relevancy >= THRESHOLD, f"🚨 REGRESSION DETECTED! Relevancy dropped to {avg_relevancy:.4f}"
        
        print("✅ System passed regression testing. Code is safe to merge into production.")