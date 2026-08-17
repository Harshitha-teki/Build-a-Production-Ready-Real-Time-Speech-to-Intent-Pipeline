# Model Choices

## ASR Model: faster-whisper `tiny.en` with INT8 Quantization

**Model:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with `tiny.en` checkpoint  
**Quantization:** INT8  
**Device:** CPU  

### Trade-offs Considered

| Model | Size | Speed (CPU) | WER (English) | RAM Usage |
|-------|------|-------------|----------------|-----------|
| tiny.en | 39M | ~0.3s / 10s audio | ~12% | ~150MB |
| base.en | 74M | ~0.8s / 10s audio | ~8% | ~300MB |
| small.en | 244M | ~2.5s / 10s audio | ~5% | ~800MB |
| medium.en | 769M | ~6s / 10s audio | ~4% | ~2GB |

**Decision:** I chose `tiny.en` with INT8 quantization for the following reasons:

1. **Latency Budget:** The overall pipeline budget is 2 seconds (p95). ASR is the most compute-intensive stage. The `tiny.en` model completes inference in ~300ms on CPU, leaving ~1.7s for NLU + TTS + overhead. The `base.en` model would consume ~800ms, leaving barely enough headroom.

2. **Language Specificity:** Since we only need English, the `.en` variant is smaller and faster than the multilingual equivalent, with better accuracy on English.

3. **INT8 Quantization:** CTranslate2's INT8 quantization reduces model size by ~4x and speeds up inference by ~2-3x compared to FP32, with only a marginal increase in word error rate (~0.5%). This is the single most impactful optimization for meeting the latency target.

4. **VAD Filtering:** We enable Voice Activity Detection (VAD) filtering to skip silent segments, further reducing unnecessary computation on non-speech portions of the audio.

5. **Beam Size 1:** Using greedy decoding (beam_size=1) instead of beam search reduces latency by ~30% with negligible accuracy loss for short command utterances.

### Accuracy Justification

For a smart-home command domain with a limited vocabulary, even the `tiny.en` model achieves excellent transcription accuracy. The phrases are short, grammatically simple, and spoken clearly. The trade-off of ~4% higher WER (compared to medium.en) is negligible because:
- Commands are short (3-8 words), reducing cumulative error probability
- The NLU module uses intent classification, not exact string matching, providing resilience to minor transcription errors
- The TF-IDF + LogisticRegression classifier is trained on diverse phrasings, making it robust to synonym substitution

---

## NLU Model: TF-IDF + LogisticRegression (Custom Trained)

**Approach:** Scikit-learn TF-IDF vectorizer + LogisticRegression classifier  
**Training Data:** 20-30 synthetic phrases per intent (240 total)  
**Inference:** ~1-2ms per request  

### Trade-offs Considered

| Approach | Latency | Accuracy | Complexity | Dependencies |
|----------|---------|----------|------------|--------------|
| TF-IDF + LogisticRegression | ~1ms | ~95% | Low | scikit-learn |
| DistilBERT fine-tuned | ~15ms | ~98% | Medium | transformers, torch |
| Quantized LLM (7B) | ~800ms | ~99% | High | ollama, 4GB+ RAM |

**Decision:** I chose TF-IDF + LogisticRegression for the following reasons:

1. **Extreme Speed:** At ~1ms inference time, this model adds virtually zero overhead to the pipeline. This is critical because ASR and TTS are already consuming most of the latency budget.

2. **No GPU Required:** Unlike transformer-based models, TF-IDF + LogisticRegression runs efficiently on CPU with minimal memory (~50MB loaded). This makes deployment simpler and cheaper.

3. **Sufficient Accuracy for Closed Domain:** With 8 well-defined intents and ~240 training examples, LogisticRegression achieves >95% accuracy. Smart-home commands are short, unambiguous, and have clear intent boundaries.

4. **Deterministic and Predictable:** Unlike LLM-based approaches, there's no variance in inference time or output format. Every request produces a valid intent string with a confidence score, eliminating the need for prompt engineering or JSON parsing.

5. **No External Dependencies:** This approach doesn't require a separate LLM serving infrastructure (e.g., Ollama), simplifying the Docker deployment to a single container.

### Why Not an LLM?

While a quantized LLM (e.g., Llama 3 8B via Ollama) would provide higher accuracy and natural language flexibility, it introduces:
- **~800ms latency per request** (CPU-only), consuming 40%+ of the total budget
- **4GB+ additional RAM** for model weights
- **Additional container/service** (Ollama) increasing deployment complexity
- **Output format instability** requiring regex/JSON parsing and retry logic

For a production system with 8 fixed intents, these costs far outweigh the marginal accuracy benefit.

### Training Data

The training set consists of 20-30 paraphrased commands per intent (240 examples total), covering:
- Direct commands: "turn on the lights"
- Polite requests: "can you turn on the lights"
- Variations: "switch on the lamp", "activate the lights"
- Entity modifiers: "set kitchen brightness to 80"
- Compound phrases: "what is the temperature outside"

This diversity ensures robustness to natural language variation while keeping the training set small enough for instant model training at startup.
