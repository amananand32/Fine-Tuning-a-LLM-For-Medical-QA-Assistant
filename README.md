Fine-Tuning a Large Language Model (LLM) for a Medical QA Assistant
1. Problem Understanding & Dataset Selection
Objective
We need to fine-tune a base LLM to function as a Medical QA Assistant. The goal is to ensure that the model provides accurate, patient-friendly medical explanations based on research papers, clinical guidelines, and trusted medical sources.
Key Challenges:
Ensuring medical accuracy and avoiding hallucinations.
Making responses understandable for non-experts.
Handling diverse question types, including symptoms, treatments, and medical conditions.
Dataset Requirements
To fine-tune an LLM for Medical QA, we need a structured, high-quality dataset containing question-answer pairs based on reliable medical literature.
Potential Datasets:
Dataset Name
Description
Source
PubMedQA
QA dataset with answers derived from PubMed research papers.
Hugging Face Datasets
MedQuAD
Medical questions from NIH and trusted sources.
NIH
LiveQA-Medical
User-generated medical questions with expert responses.
TREC Challenge
MIMIC-III
Clinical notes and patient records.
PhysioNet

Primary Choice: PubMedQA
Contains structured question-answer pairs based on real research papers.
Well-suited for LLM fine-tuning.
Available via Hugging Face Datasets, simplifying data loading.
Data Collection & Quality Control
Remove duplicates and irrelevant data.
Ensure dataset diversity to avoid biases.
Use human annotators to validate the quality of simplified explanations.

2. Model Selection & Architecture Considerations
Base Model Choice
We need a lightweight yet powerful LLM.
Model
Size
Pros
Cons
GPT-3.5
~175B
Highly capable
Requires API, costly
LLaMA-2 7B
7B
Open-source, efficient
Needs adaptation
Mistral 7B
7B
Strong performance
Relatively new
DistilGPT-2
~124M
Small, fast
Limited medical knowledge

Choice: DistilGPT-2 + LoRA
Lightweight → Easier to fine-tune with limited computing.
Supports LoRA fine-tuning, reducing memory overhead.
Can be augmented with retrieval-based QA for better accuracy.
Fine-Tuning Strategy
Approach
Pros
Cons
When to Use?
Full Fine-Tuning
Improves model knowledge deeply
Expensive, requires more compute
If retraining from scratch
LoRA (Low-Rank Adaptation)
Memory-efficient, fast
Less adaptable to major domain shifts
If working with limited compute
PEFT (Parameter Efficient Fine-Tuning)
Similar to LoRA, efficient
May have limitations for drastic changes
When fine-tuning a large model

Why LoRA?
Allows fine-tuning only for a small number of model parameters.
Reduces VRAM usage, making it feasible on consumer GPUs.
Widely supported in Hugging Face Transformers.
Handling Token Limits & Memory Constraints
Use a max token length of 512 to handle long medical texts.
Employ gradient checkpointing to reduce memory usage.
Use mixed precision training (FP16) to optimize memory consumption.
Handling inference efficiency
QLORA -  Quantize the model (4-bit or 8-bit).
Prune unnecessary weights or neurons.
Use distilled or smaller models.
Batch inputs for parallel processing.
Use caching and Flash Attention.
Precompute embeddings for reusable inputs.


3. Fine-Tuning Strategy
Fine-tuning an LLM effectively requires careful selection of hyperparameters to balance accuracy, training time, and computational cost.
Key Hyperparameters
Hyperparameter
Suggested Value
Reasoning
Batch Size
8 (or 16 if memory allows)
Keeps memory usage low while allowing stable training.
Learning Rate
5e-5
Prevents catastrophic forgetting.
Epochs
3-5
Prevents overfitting while ensuring knowledge adaptation.
Weight Decay
0.01
Helps prevent overfitting.
Gradient Accumulation Steps
4
Helps simulate larger batch sizes on limited hardware.
Max Token Length
512
Ensures model can process longer questions/answers.
Optimizer
AdamW
Well-suited for transformer-based models.

Choosing a Loss Function
For Medical QA, the ideal loss function depends on the task type:
For Generative QA (Text Generation) → Use CrossEntropyLoss
Suitable for Causal Language Models (CLMs) like GPT-2.
Helps the model learn better text generation by minimizing token prediction errors.
For Extractive QA (Span-based Answers) → Use Span-based Loss (not needed here).
Choice: CrossEntropyLoss (default in Hugging Face Trainer for causal models)

Supervised Fine-Tuning vs. Reinforcement Learning for Medical QA on PubMedQA
Why Supervised Fine-Tuning (SFT) is the Best Approach?
 Uses a labeled dataset → We already have question-answer pairs from PubMedQA.
 Proven for QA tasks → Works well for medical question answering.
 Less expensive → RLHF requires human labeling of rewards, which is costly.
 Avoids instability → RLHF can cause unpredictable behavior in sensitive medical applications.
Final Decision: Use Supervised Fine-Tuning for PubMedQA. RLHF is unnecessary for initial training but could be explored later for refinement.
Avoiding Catastrophic Forgetting & Preserving General Knowledge
Strategies to prevent forgetting:
Mixing dataset types: Combine PubMedQA with general medical texts to retain broad knowledge.
Lower learning rate: Prevents drastic weight changes that overwrite previous knowledge.
LoRA-based fine-tuning: Keeps original model weights mostly intact.
Regularization (EWC or L2 weight decay): Encourages weights to stay close to their original values.

4. Evaluation & Validation
Key Metrics to Track
Metric
Use Case
Description
Perplexity (PPL)
Language fluency
Measures next-word prediction accuracy (lower   = better).
BLEU Score
Text similarity
Measures overlap between model-generated and reference answers.
ROUGE Score
Summarization/QA
Evaluates recall of key words from reference 
answers.
Medical Expert Evaluation


Accuracy
Human doctors evaluate correctness of answers.

Evaluation Strategy
We compute BLEU, ROUGE, and Perplexity scores to compare model-generated answers with ground-truth answers.
 Steps:
Take test set questions from PubMedQA.
Generate answers using the fine-tuned model.
Compare generated answers with ground-truth answers using ROUGE & BLEU.
Calculate Perplexity (PPL) to measure fluency.
Best Evaluation Approach for PubMedQA:
Use ROUGE & BLEU for automatic evaluation.
Use human experts to verify accuracy & hallucinations.
If extractive QA is needed, F1 Score is useful.
Analysis Of Results Before and After Finetuning:
Perplexity  Before: 289.58  and After:125.88
BLEU Score  Before: 0.0167 and After: 0.0263
ROUGE-1  Before: 0.249 and After: 0.184
ROUGE-2  Before: 0.100 and After: 0.053
5. Deployment & Maintenance
Monitoring Post-Deployment
Use logging and monitoring tools for real-time tracking.
Set up alerts for model drift.
Handling Model Drift
Regularly update the model with new medical research data.
Use active learning to improve based on user feedback.
Cost Optimization
Use model quantization to reduce inference costs.
Deploy the model on serverless infrastructure to scale based on demand.



