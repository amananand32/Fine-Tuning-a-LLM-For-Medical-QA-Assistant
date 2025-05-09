# Fine-Tuning a Large Language Model (LLM) for a Medical QA Assistant

This project demonstrates how to fine-tune a base LLM to function as a Medical QA Assistant, providing accurate, patient-friendly explanations based on trusted medical sources.

---

## 1. Problem Understanding & Dataset Selection

- **Objective:**  
  Fine-tune a base LLM to provide accurate, patient-friendly medical explanations using research papers, clinical guidelines, and trusted sources.

- **Key Challenges:**  
  - Ensuring medical accuracy and avoiding hallucinations  
  - Making responses understandable for non-experts  
  - Handling diverse question types: symptoms, treatments, conditions

- **Dataset Requirements:**  
  Structured, high-quality question-answer pairs from reliable medical literature.

- **Potential Datasets:**  
  | Dataset         | Description                                      | Source                  |
  |-----------------|--------------------------------------------------|-------------------------|
  | PubMedQA        | QA pairs from PubMed research papers             | Hugging Face Datasets   |
  | MedQuAD         | Medical questions from NIH & trusted sources     | NIH                     |
  | LiveQA-Medical  | User-generated questions with expert answers     | TREC Challenge          |
  | MIMIC-III       | Clinical notes and patient records               | PhysioNet               |

- **Primary Choice:** PubMedQA  
  - Structured QA pairs based on real research papers  
  - Well-suited for LLM fine-tuning  
  - Available via Hugging Face Datasets

- **Data Quality Control:**  
  - Remove duplicates and irrelevant data  
  - Ensure dataset diversity to avoid bias  
  - Use human annotators to validate explanations

---

## 2. Model Selection & Architecture Considerations

- **Base Model Options:**  
  | Model        | Size   | Pros                   | Cons                     |
  |--------------|--------|------------------------|--------------------------|
  | GPT-3.5      | ~175B  | Highly capable         | API only, costly         |
  | LLaMA-2 7B   | 7B     | Open-source, efficient | Needs adaptation         |
  | Mistral 7B   | 7B     | Strong performance     | Relatively new           |
  | DistilGPT-2  | ~124M  | Small, fast            | Limited medical knowledge|

- **Chosen Model:** DistilGPT-2 + LoRA  
  - Lightweight and easy to fine-tune on limited hardware  
  - Supports LoRA for memory-efficient fine-tuning  
  - Can be combined with retrieval-based QA for improved accuracy

- **Fine-Tuning Strategies:**  
  | Approach               | Pros                      | Cons                           | Use Case                  |
  |------------------------|---------------------------|--------------------------------|---------------------------|
  | Full Fine-Tuning       | Deep knowledge improvement| Expensive, high compute        | Retraining from scratch   |
  | LoRA (Low-Rank Adapt.) | Memory-efficient, fast    | Less adaptable to big shifts   | Limited compute           |
  | PEFT                   | Efficient, similar to LoRA| May limit drastic changes      | Large model fine-tuning   |

- **Why LoRA?**  
  - Fine-tunes fewer parameters, reducing VRAM usage  
  - Supported in Hugging Face Transformers

- **Token & Memory Constraints:**  
  - Max token length: 512  
  - Use gradient checkpointing and mixed precision (FP16)  
  - Quantize model with QLoRA (4/8-bit)  
  - Prune unnecessary weights  
  - Batch inputs, use caching/Flash Attention

---

## 3. Fine-Tuning Strategy

- **Key Hyperparameters:**  
  | Hyperparameter          | Suggested Value | Reasoning                       |
  |------------------------|-----------------|---------------------------------|
  | Batch Size             | 8 (or 16)       | Balances memory and stability   |
  | Learning Rate          | 5e-5            | Prevents catastrophic forgetting|
  | Epochs                 | 3-5             | Avoids overfitting, adapts model|
  | Weight Decay           | 0.01            | Regularization                  |
  | Gradient Accumulation  | 4               | Simulates larger batch size     |
  | Max Token Length       | 512             | Handles longer texts            |
  | Optimizer              | AdamW           | Suited for transformers         |

- **Loss Function:**  
  - Use `CrossEntropyLoss` for generative QA (default for causal models like GPT-2)

- **Supervised Fine-Tuning (SFT) vs. RLHF:**  
  - SFT preferred due to labeled QA pairs from PubMedQA  
  - SFT is proven, less expensive, and more stable for medical QA  
  - RLHF can be explored later for refinement

- **Avoiding Catastrophic Forgetting:**  
  - Mix PubMedQA with general medical texts  
  - Use lower learning rates  
  - Apply LoRA fine-tuning to preserve base weights  
  - Employ regularization (EWC or L2 weight decay)

---

## 4. Evaluation & Validation

- **Key Metrics:**  
  | Metric       | Use Case          | Description                                    |
  |--------------|-------------------|------------------------------------------------|
  | Perplexity   | Language fluency  | Lower is better; next-word prediction accuracy |
  | BLEU Score   | Text similarity   | Overlap with reference answers                 |
  | ROUGE Score  | Summarization/QA  | Recall of key words                            |
  | Expert Eval  | Medical accuracy  | Human doctors assess answers                   |

- **Evaluation Steps:**  
  - Use PubMedQA test set questions  
  - Generate answers with fine-tuned model  
  - Compare using BLEU, ROUGE, and Perplexity  
  - Validate with human experts for accuracy and hallucination detection

- **Results (Before vs After Fine-Tuning):**  
  - Perplexity: 289.58 → 125.88  
  - BLEU Score: 0.0167 → 0.0263  
  - ROUGE-1: 0.249 → 0.184  
  - ROUGE-2: 0.100 → 0.053

---

## 5. Deployment & Maintenance

- **Monitoring:**  
  - Use logging and monitoring tools for real-time tracking  
  - Set alerts for model drift

- **Handling Model Drift:**  
  - Regularly update with new medical research data  
  - Use active learning based on user feedback

- **Cost Optimization:**  
  - Apply model quantization to reduce inference costs  
  - Deploy on serverless infrastructure for scalable demand-based usage

---

## License

[Add your license here]

---

## Citation

If you use this work, please cite or reference this repository.

