## Running Instructions:

1. run:
   pip install -r requirements.txt
2. Input newline-separated sentences into input_mh.txt
3. run:
   ./entry_point.sh

   This will automatically execute ollama.sh, which starts an ollama server in the background.
   The default model is llama3.1:8b. If you want to use a different model, edit the entry_point.sh
   script to pull the model you'd like to use. Also, you will need to change the 'model_type' parameter in the metaphor_handler_batch_word_level.py main function.
4. Translated sentences are written to output_mh.txt



## Parameters

All parameters are set to the optimized values described in Chapter 4 of the thesis. See `main()` in `metaphor_handler_batch_word_level.py` for the exact configurations used.

    model_type: str = 'llama3.1:8b'
        Ollama model identifier. Reference the Ollama documentation for available models.

    similarity_function: str = 'bleu'
        Determines which similarity metric to use for comparing the candidate with the original sentence.
        Options: "rouge", "bleu", or "bert_cosine".

    bert_embedder: BertEmbedder | None = None
        Required only if using "bert_cosine" as the similarity function.
        When set to None, the translator will automatically use
        the "sentence-transformers/all-MiniLM-L6-v2" embedder.

    md: MetaphorDetector | None = None
        Allows passing in a pre-initialized MetaphorDetector instance.
        Speeds up runtime if multiple translator instances are created.

    start_temp: float = 0.2
        Temperature parameter for the Ollama LLM.
        Higher temperatures produce more varied (creative) outputs.

    desired_sim_score: float = 0.1
        Similarity score threshold that determines whether a candidate passes
        the similarity criterion.

    prompt_limit: int = 4
        Maximum number of prompt batches generated per input sentence.

    batch_size: int = 10
        Number of candidate translations generated per batch.

    reduction_rate: float = 0.82
        Controls metaphor reduction aggressiveness.
        See Chapter 3 of the thesis for a detailed explanation.
        Used to compute the maximum allowable residual metaphors
        for a candidate to be considered acceptable.

## Terminal output:

The terminal first outputs the sentence being translated, followed by a stream of symbols.
If no symbols follow, this simply means that no metaphors were detected in the original sentence, so the sentence bypasses the metaphor translator completely (the output is identical to the input).

Otherwise each symbol, for a single candidate translation, holistically represents both sentence similarity and metaphor re-detection results. The keys for each symbol
are below:

✔   Residual metaphors less than max. allowed (good), but does not meet sentence similarity criteria (bad)
•   Sentence similarity passes (good), but residual metaphors greater than max. allowed (bad)
.   Residual metaphors greater than max. allowed (bad), and does not meet sentence similarity criteria (bad)
★   Both sentence similarity and metaphor redetection criteria pass (good, triggers early stopping)

The symbols are seperated by batch with a space, as in the example below:

••.•.• ••...★

This example shows that two batches were produced, and since early stopping was triggered (at least one ★ candidate was generated), a third batch was not necessary.

The second example below shows three batches of 6 candidates each, where early stopping was not activated.

...... .✔.✔.. ...✔.✔






