AI Disclosure

Name: Mahammad Niftullayev
Tools used: ChatGPT (OpenAI) and Claude (Anthropic)

During this project I used AI assistants for understanding concepts, debugging, reviewing code, and improving the implementation. Claude was used mainly for the final version of generate.py and for checking the report numbers; ChatGPT was used for the other parts, as described below.

Part A – Tokenizer: debugging. I implemented the byte-level BPE tokenizer and used ChatGPT to understand and investigate the indexing error described in R1. I also had the tie rule and the round-trip checked with code review.
Part B – Model: code review. I studied the Transformer architecture and implemented the model with manual attention. I used ChatGPT and Claude to review the model structure against the specification and to understand implementation details.
Part C – Training pipeline: debugging. I developed the training pipeline, the bigram baseline and the configs, and ran every training run myself. I used ChatGPT to investigate problems and Claude to review the schedule, clipping and logging against the specification.
Part D – Inference: code generation. Claude wrote the final version of generate.py so that it matches the required interface (list in and list out, temperature=0 as greedy, top-k then top-p, seed). I tested it myself: I compared cached and uncached greedy outputs, ran the speed benchmark on my GPU, and generated the samples for R4.
Part E – Experiments: concepts. I chose and ran the experiments, collected the results, and compared model performance. I used AI assistance to interpret the findings.
Part F – Report: code review. I wrote the reflection answers from my own runs and commits. Claude checked every number against runs.csv, the logs and samples.txt, and AI assistance was used to organize the results and improve the clarity of my explanations.

I was responsible for running the code, conducting the experiments, checking the results, and understanding the implementation. I can explain every submitted file line by line.