# Awesome-Summarization-LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME things about **Long-Context Comprehension and Summarization Based on LLMs**.

Large Language Models (LLMs) have made remarkable progress in natural language processing tasks. However, they often struggle to handle large amounts of input context, such as books and movie scripts, which can lead to redundant, inaccurate or incoherent summaries. This repository aims to promote research in this area by providing a collated list of research papers.


## Table of Contents

- [Awesome-Summarization-LLM ](#awesome-summarization-llm-)
  - [Table of Contents](#table-of-contents)
  - [Datasets, Benchmarks \& Surveys](#datasets-benchmarks--surveys)
  - [Method](#method)
  - [evaluate](#evaluate)
  - [Applications](#applications)
    - [Knowledge Graph](#knowledge-graph)
    - [Others](#others)
  - [Resources \& Tools](#resources--tools)
  - [Contributing](#contributing)
  - [Star History](#star-history)
 

## Datasets, Benchmarks & Surveys
- (*NAACL'21*) Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training [[paper](https://aclanthology.org/2021.naacl-main.278/)][[code](https://github.com/google-research-datasets/KELM-corpus)]
- (*ACL'22*) SummScreen: A Dataset for Abstractive Screenplay Summarization[[paper](https://aclanthology.org/2022.acl-long.589/)][[code](https://github.com/mingdachen/SummScreen)]
- (*ACL'23*) MeetingBank: A Benchmark Dataset for Meeting Summarization[[paper](https://aclanthology.org/2023.acl-long.906/)][[code](https://meetingbank.github.io/dataset/)]
- (*ACL'24*) LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding[[paper](https://arxiv.org/abs/2308.14508)][[code](https://github.com/THUDM/LongBench)]
- (*EMNLP'22*) BookSum: A Collection of Datasets for Long-form Narrative Summarization[[paper](https://aclanthology.org/2022.findings-emnlp.488/)][[code](https://github.com/salesforce/booksum)]
- (*NAACL'24*) Select and Summarize: Scene Saliency for Movie Script Summarization[[paper](https://aclanthology.org/2024.findings-naacl.218/)][[code](https://github.com/saxenarohit/select_summ)]
- (*LREC'24*) MDS: A Fine-Grained Dataset for Multi-Modal Dialogue Summarization[[paper](https://aclanthology.org/2024.lrec-main.970/)][[code](https://github.com/R00kkie/MDS)]
- (*arXiv 2024.06*) SUMIE: A Synthetic Benchmark for Incremental Entity Summarization [[paper](https://arxiv.org/abs/2406.05079)][[code](https://github.com/google-research-datasets/sumie)]  
- (*arXiv 2024.06*) A Systematic Survey of Text Summarization: From Statistical Methods to Large Language Models[[paper](https://arxiv.org/abs/2406.11289)]
- (*arXiv 2024.07*) Retrieval-Augmented Generation for Natural Language Processing: A Survey[[paper](https://arxiv.org/abs/2407.13193)][[code]()]

## Method
- (*EMNLP'23*) StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper](https://arxiv.org/abs/2305.09645)][[code](https://github.com/RUCAIBox/StructGPT)]
- (*NAACL'24*) Select and Summarize: Scene Saliency for Movie Script Summarization[[paper](https://aclanthology.org/2024.findings-naacl.218/)][[code](https://github.com/saxenarohit/select_summ)]
- (*ACL'24*) On Context Utilization in Summarization with Large Language Models[[paper](https://arxiv.org/abs/2310.10570)][[code](https://github.com/ntunlp/MiddleSum)]
- (*AAAI'24*) Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper](https://arxiv.org/abs/2308.09687)][[code](https://github.com/spcl/graph-of-thoughts)]
- (*arXiv 2024.01*) GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models[[paper](https://arxiv.org/abs/2406.14550)][[code]()]
- (*arXiv 2024.03*) Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers[[paper](https://arxiv.org/abs/2403.01061)][[code](https://github.com/melaniesubbiah/reading-subtext)]
- (*arXiv 2024.06*) Chain of Agents: Large Language Models Collaborating on Long-Context Tasks[[paper](https://arxiv.org/abs/2406.02818)][[code]()]
- (*arXiv 2024.07*) Enhancing Incremental Summarization with Structured Representations[[paper](https://arxiv.org/abs/2407.15021v1)][[code]()]
  
## evaluate
- (*ICLR'2024*) BooookScore: A systematic exploration of book-length summarization in the era of LLMs[[paper](https://arxiv.org/abs/2310.00785)][[code](https://github.com/lilakk/BooookScore)]
- (*arXiv 2024.04*) FABLES: Evaluating faithfulness and content selection in book-length summarization[[paper](https://arxiv.org/abs/2404.01261)][[code](https://github.com/mungg/FABLES)]
- (*arXiv 2024.04*) Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works[[code](https://arxiv.org/abs/2404.12726)][[code](https://github.com/Joanna0123/character_profiling)]

## Applications

### Knowledge Graph
- (*AAAI'22*) Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21286)]
- (*EMNLP'22*) Language Models of Code are Few-Shot Commonsense Learners [[paper](https://arxiv.org/abs/2210.07128)][[code](https://github.com/reasoning-machines/CoCoGen)]
- (*SIGIR'23*) Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction [[paper](https://arxiv.org/abs/2210.10709)][[code](https://github.com/zjunlp/RAP)]
- (*TKDE'23*) AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models [[paper](https://arxiv.org/abs/2307.11772)][[code](https://github.com/ruizhang-ai/AutoAlign)]
- (*AAAI'24*) Graph Neural Prompting with Large Language Models [[paper](https://arxiv.org/abs/2309.15427)][[code](https://github.com/meettyj/GNP)]
- (*NAACL'24*) zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*ICLR'24*) Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph [[paper](https://arxiv.org/abs/2307.07697)][[code](https://github.com/IDEA-FinAI/ToG)]
- (*arXiv 2023.04*) CodeKGC: Code Language Model for Generative Knowledge Graph Construction [[paper](https://arxiv.org/abs/2304.09048)][[code](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC)]
- (*arXiv 2023.05*) Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs [[paper](https://arxiv.org/abs/2305.09858)]
- (*arXiv 2023.08*) MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models [[paper](https://arxiv.org/abs/2308.09729)][[code](https://github.com/wyl-willing/MindMap)]
- (*arXiv 2023.10*) Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph [[paper](https://arxiv.org/abs/2310.16452)]
- (*arXiv 2023.10*) Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning [[paper](https://arxiv.org/abs/2310.01061)][[code](https://github.com/RManLuo/reasoning-on-graphs)]
- (*arXiv 2023.11*) Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*arXiv 2023.12*) KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn’t Know [[paper](https://arxiv.org/abs/2312.11539)]
- (*arXiv 2024.02*) Large Language Model Meets Graph Neural Network in Knowledge Distillation [[paper](https://arxiv.org/abs/2402.05894)]
- (*arXiv 2024.02*) Large Language Models Can Learn Temporal Reasoning [[paper](https://arxiv.org/pdf/2401.06853v2.pdf)][[code](https://github.com/xiongsiheng/TG-LLM)]
- (*arXiv 2024.03*) Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments [[paper](https://arxiv.org/abs/2403.08593)]
- (*arXiv 2024.04*) Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs [[paper](https://arxiv.org/abs/2404.00942)][[code](https://github.com/xz-liu/GraphEval)]
- (*arXiv 2024.04*) Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction [[paper](https://arxiv.org/abs/2404.03868)]
- (*arXiv 2024.05*) FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering [[paper](https://arxiv.org/abs/2405.13873)]
- (*arXiv 2024.06*) Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph [[paper](https://arxiv.org/abs/2406.01145)]
  
### Others
- (*arXiv 2024.07*) Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems[[paper](https://arxiv.org/abs/2407.01370)][[code](https://github.com/salesforce/summary-of-a-haystack)]


## Resources & Tools
- [GraphGPT: Extrapolating knowledge graphs from unstructured text using GPT-3](https://github.com/varunshenoy/GraphGPT)


## Contributing
👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Lemonnn7/Awesome-Summarization-LLM&type=Date)](https://star-history.com/#Lemonnn7/Awesome-Summarization-LLM&Date)
