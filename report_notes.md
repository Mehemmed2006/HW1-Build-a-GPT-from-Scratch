Part F – Reflection Answers

All numbers come from runs.csv, logs/<run_id>.csv, samples.txt and the tokenizer file. Training loss in the logs is the loss of a single batch, so it is noisy. Validation loss is the mean over 50 random validation batches.

### R1. Which bug cost you the most time?

**Symptom:** While training the byte-level BPE tokenizer, I encountered the following error:

```text
IndexError: index 978850 is out of bounds for axis 0 with size 978844
```

**Cause:** The merge operation used pair positions calculated before modifying the token sequence. When a merge removed elements, the sequence became shorter, but the previously stored positions remained unchanged. As a result, some positions pointed outside the sequence boundaries.

**How I found it:** I investigated how the sequence length changed during merging and compared the stored pair positions with the updated sequence length. This helped me identify that the problem was caused by using outdated positions after modifying the sequence.

**Fix:** I rewrote the merge operation to scan the current sequence once, from left to right, and replace non-overlapping occurrences as it goes. This avoids using outdated positions and prevents the index error.

The initial byte-level BPE tokenizer implementation is recorded in commit `4248f23` (Implement byte-level BPE tokenizer). The bug investigation and documentation are recorded in commits `bc26726` and `37c44d7`.

R2. Which experiment result surprised you most?

Removing dropout (A4) changed the validation loss almost not at all, while the training loss moved a lot.

Run	final_train_loss	best_val_loss	train–val gap at step 3000
M (dropout 0.1)	3.5255	3.7745	0.249
A4 (dropout 0.0)	2.8950	3.7708	0.876

The validation difference is only 0.0037, which is within the noise of a single seed and a 50-batch estimate. The logs show that A4 learned faster early (validation 3.8380 at step 1500 against 3.9909 for M, logs/A4.csv and logs/M.csv), but M caught up by step 3000. My explanation: without dropout the model fits the training text much more closely (the gap is 3.5 times larger), but on this small corpus and in 3000 steps this does not translate into better validation loss. I did not test longer training, where the gap might matter more. Runs: M, A4.

R3. Which single change had the largest effect on validation loss?

Among the single-factor ablations, removing the position embedding (A1) had the largest effect: best validation loss rose from 3.7745 (M) to 4.0141 (A1), an increase of 0.2396. Without positions the model only sees token content and has no direct order information, even though the causal mask still gives it some implicit order.

For comparison, the scaling runs changed more: S (4.2735) against M (3.7745) differs by 0.4988, but S changes depth, width and head count at once, so I do not count it as a single change. The other ablations were smaller: A2 +0.0518, A3 −0.0707, A4 −0.0037, A5 −0.1003 (A5 also saw twice as many tokens). Compared runs: M and A1.

R4. Three samples of 120 tokens from run L at temperature 0.8

I generated 3 samples from checkpoints/L_best.pt (run L, 2,880,384 parameters) with prompt ROMEO:, 120 new tokens, temperature 0.8, seed 1, without top-k or top-p. The samples are in samples.txt.

Failure pattern: invented words. The model gets the play format right (speaker labels such as Nurse: and JULIET:, short verse lines), but it produces non-words built from plausible fragments: infurthorse and particient (sample 1), nuptedious and househood (sample 2), askied and provate (sample 3).

Connection to a design choice. I think this is related to the tokenizer: with vocab_size = 1024 and 2.39 bytes per token (R5), most words are split into several fragments, and the model can combine fragments into strings that never occur in the text. Model size and data also matter: logs/L.csv shows training loss 2.9025 against validation loss 3.6629 at step 3000, a gap of 0.760 (0.249 for M), so L fits the small corpus but does not generalize cleanly. I did not run an experiment that separates these causes, so this is a hypothesis.

R5. First 10 merges and bytes per token
Merge	Token id	Pair	Text
0	256	(101, 32)	e 
1	257	(116, 104)	th
2	258	(116, 32)	t 
3	259	(115, 32)	s 
4	260	(100, 32)	d 
5	261	(44, 32)	, 
6	262	(111, 117)	ou
7	263	(101, 114)	er
8	264	(105, 110)	in
9	265	(121, 32)	y 

The validation text has 111,540 bytes and 46,683 tokens, so the tokenizer gives 111,540 / 46,683 = 2.3893 bytes per token.

Each merge is the most frequent adjacent pair at that moment in the training text. Six of the ten merges end in a space (e , t , s , d , , , y ): because there is no pre-splitting into words, a letter at the end of a word can merge with the following space, and in English e, t, s, d, y are very common word endings, while a comma is almost always followed by a space. The other four (th, ou, er, in) are among the most common letter pairs in English.