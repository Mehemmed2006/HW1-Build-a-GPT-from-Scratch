## R1. Which bug cost you the most time?

The bug that cost me the most time occurred during the implementation of the byte-level BPE tokenizer.

The main symptom was an `IndexError` during the merge process:

`IndexError: index 978850 is out of bounds for axis 0 with size 978844`

The problem occurred because token-pair positions became invalid after elements were removed from the sequence. The merge operation was using positions calculated before the sequence was modified.

I identified the issue by examining the indexing logic and checking how the sequence length changed after each merge. I fixed the position handling by adjusting the remaining positions after deletions and ensuring that the merge operation used valid indices.

The related commit is **`4248f23` – Day 2: Implement byte-level BPE tokenizer**.

I followed a commit policy in which I committed the completed work at the end of each day. I did not create a commit while known bugs or incomplete tasks remained. Therefore, the bug fix was included in the general Day 2 commit rather than being recorded in a separate bug-fix commit.



## R2. Which experiment result surprised you the most?

The experiment that surprised me the most was A4, in which dropout was removed from the model.

The M baseline achieved a final training loss of **3.5255** and a best holdout shard loss of **3.7745**. In comparison, A4 achieved a much lower final training loss of **2.8950**, while its best holdout shard loss was **3.7708**.

This result was surprising because removing dropout significantly improved the training loss, but it barely changed the holdout shard loss. The difference in holdout shard loss was only **0.0037**.

This suggests that the model fitted the training data more closely without dropout, but this did not lead to a meaningful improvement in performance on unseen data.

**Relevant runs:** `M` and `A4` (A4: `A4_no_dropout`).


## R3. Which single change had the largest effect on validation loss?

The removal of positional embeddings (A1) had the largest negative effect on the holdout shard loss among the controlled ablation experiments.

The baseline model M achieved a best holdout shard loss of **3.7745**, while A1 achieved **4.0141**. This represents an increase of approximately **0.2396** in loss.

This result indicates that positional embeddings play an important role in the model's ability to understand token order and position. Without positional information, the model performed worse on the holdout shard.

**Compared runs:** `M` and `A1` (`A1_no_positional_embeddings`).



## R4. Generate 3 samples of 120 tokens from run L at temperature 0.8. Describe one failure pattern and connect it to a design choice.

I generated three text samples using the GPT-L model (2,880,384 parameters) with the prompt "ROMEO:". Each sample contained 120 newly generated tokens. I used top-p sampling with a temperature of 0.8 and a top-p value of 0.9.

One noticeable failure pattern was grammatical and semantic inconsistency. For example, the generated text contained phrases such as "By no honour of your appetition" and "a gentle dancture". Although the model learned Shakespeare-like vocabulary, character names, and dialogue structures, some sentences were difficult to understand.

This limitation is related to the model's relatively small architecture and limited training data. The model learned local language patterns but struggled to maintain grammatical accuracy and coherent meaning over longer passages. Top-p sampling also introduced variation in the generated text.

**Model:** L  
**Prompt:** ROMEO:  
**Generation settings:** 120 tokens per sample, temperature 0.8, top-p 0.9  
**Output file:** samples.txt


## R5. List the first 10 merges as text and bytes-per-token on the validation text. Why do the first merges look like that?

The first 10 BPE merges learned by my tokenizer were:

| Merge | Token IDs | Merged text | Bytes (hex) |
|---:|---|---|---|
| 1 | (101, 32) | `e ` | `65 20` |
| 2 | (116, 104) | `th` | `74 68` |
| 3 | (116, 32) | `t ` | `74 20` |
| 4 | (115, 32) | `s ` | `73 20` |
| 5 | (100, 32) | `d ` | `64 20` |
| 6 | (44, 32) | `, ` | `2C 20` |
| 7 | (111, 117) | `ou` | `6F 75` |
| 8 | (101, 114) | `er` | `65 72` |
| 9 | (105, 110) | `in` | `69 6E` |
| 10 | (121, 32) | `y ` | `79 20` |

The holdout shard contained **111,540 bytes** and **46,683 tokens**.

Bytes per token:

**111,540 / 46,683 = 2.3893 bytes per token.**

The first merges look like this because BPE selects the most frequent adjacent byte pairs in the training text. Common English letter combinations, such as `th`, `ou`, and `er`, occur frequently. Pairs containing spaces, such as `e ` and `t `, are also common because they frequently appear at word boundaries.

These merges show that the tokenizer learns frequent patterns directly from the text without requiring predefined word boundaries.