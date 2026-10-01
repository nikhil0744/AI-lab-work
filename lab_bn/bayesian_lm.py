"""
Bayesian Language Models: First-Order and Second-Order Autoregressive Models.

This module implements:
1. First-Order Autoregressive Bayesian Language Model: P(Xt | Xt-1)
2. Second-Order Autoregressive Bayesian Language Model: P(Xt | Xt-2, Xt-1)
3. CPT inspection, validation, and probability normalisation verification
4. Next-word prediction (argmax and probability distribution)
5. Text generation in both Greedy (Mode A) and Sampling (Mode B) modes
6. Model comparison metrics (distinct parameters, zero-probability contexts, diversity, coherence)
"""

import sys
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any


# Standard laboratory training corpus
DEFAULT_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START_TOKEN = "<START>"
END_TOKEN = "<END>"


class FirstOrderBayesianLM:
    """
    First-order autoregressive Bayesian Language Model.
    Models the joint sequence distribution via the first-order Markov assumption:
        P(X1, X2, ..., XT) = P(X1 | <START>) * prod_{t=2}^T P(Xt | Xt-1)
    where:
        P(Xt = wj | Xt-1 = wi) = C(wi, wj) / sum_k C(wi, wk)
    """

    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus = corpus or DEFAULT_CORPUS
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.tokenized_sentences: List[List[str]] = []
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.cpt: Dict[str, Dict[str, float]] = {}
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: List[str]) -> None:
        """Tokenize corpus, count bigram transitions, and compute CPT."""
        self.corpus = corpus
        self.transition_counts.clear()
        self.cpt.clear()

        # Tokenize with <START> and <END>
        self.tokenized_sentences = []
        all_tokens = set()
        for sentence in self.corpus:
            tokens = [START_TOKEN] + sentence.lower().strip().split() + [END_TOKEN]
            self.tokenized_sentences.append(tokens)
            all_tokens.update(tokens)

        self.vocab = sorted(list(all_tokens))
        self.vocab_size = len(self.vocab)

        # Count bigram transitions C(wi, wj)
        for tokens in self.tokenized_sentences:
            for w1, w2 in zip(tokens[:-1], tokens[1:]):
                self.transition_counts[w1][w2] += 1

        # Compute Conditional Probability Table P(Xt = wj | Xt-1 = wi)
        for w1, counts in self.transition_counts.items():
            total = sum(counts.values())
            self.cpt[w1] = {w2: count / total for w2, count in counts.items()}

    def get_cpt(self, context: str) -> Dict[str, float]:
        """Return the probability distribution P(Xt | Xt-1 = context)."""
        return self.cpt.get(context, {})

    def check_normalization(self, tolerance: float = 1e-6) -> Dict[str, Tuple[float, bool]]:
        """
        Verify that for every observed context word w:
            sum_v P(v | w) == 1.0
        """
        results = {}
        for word, dist in self.cpt.items():
            total = sum(dist.values())
            is_valid = abs(total - 1.0) < tolerance
            results[word] = (total, is_valid)
        return results

    def predict_next(self, context: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Return (argmax_word, max_prob, distribution) for a given context.
        If context is unknown, returns (None, 0.0, {}).
        """
        dist = self.get_cpt(context)
        if not dist:
            return None, 0.0, {}
        # Deterministic tie-breaking: highest probability, then alphabetical
        sorted_candidates = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        best_word, best_prob = sorted_candidates[0]
        return best_word, best_prob, dist

    def generate_sentence(
        self,
        mode: str = "sampling",
        max_length: int = 25,
        seed: Optional[int] = None,
        tie_break: str = "alpha",
    ) -> List[str]:
        """
        Generate a sentence using ancestral sampling or greedy argmax.
        Modes:
            - 'greedy': Always chooses argmax_w P(w | w_prev)
            - 'sampling': Samples w ~ P(w | w_prev)
        Stops when END_TOKEN is reached or max_length exceeded.
        """
        rng = random.Random(seed) if seed is not None else random
        tokens = [START_TOKEN]

        while tokens[-1] != END_TOKEN and len(tokens) < max_length:
            curr = tokens[-1]
            dist = self.get_cpt(curr)
            if not dist:
                # Dead end (e.g. unobserved or absorbing state)
                break

            if mode == "greedy":
                if tie_break == "alpha":
                    sorted_cand = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
                    next_word = sorted_cand[0][0]
                else:
                    max_p = max(dist.values())
                    best_words = [w for w, p in dist.items() if p == max_p]
                    next_word = best_words[0]
            elif mode == "sampling":
                words, probs = zip(*dist.items())
                next_word = rng.choices(words, weights=probs, k=1)[0]
            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            tokens.append(next_word)

        return tokens


class SecondOrderBayesianLM:
    """
    Second-order autoregressive Bayesian Language Model.
    Models the joint sequence distribution via the second-order Markov assumption:
        P(X1, X2, ..., XT) = P(X1 | <START>, <START>) * P(X2 | <START>, X1) * prod_{t=3}^T P(Xt | Xt-2, Xt-1)
    where:
        P(Xt = wk | Xt-2 = wi, Xt-1 = wj) = C(wi, wj, wk) / sum_m C(wi, wj, wm)
    """

    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus = corpus or DEFAULT_CORPUS
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.tokenized_sentences: List[List[str]] = []
        self.transition_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.cpt: Dict[Tuple[str, str], Dict[str, float]] = {}
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: List[str]) -> None:
        """Tokenize corpus with 2 <START> tokens, count trigrams, and compute CPT."""
        self.corpus = corpus
        self.transition_counts.clear()
        self.cpt.clear()

        # Tokenize with two <START> boundary tokens and one <END>
        self.tokenized_sentences = []
        all_tokens = set()
        for sentence in self.corpus:
            tokens = [START_TOKEN, START_TOKEN] + sentence.lower().strip().split() + [END_TOKEN]
            self.tokenized_sentences.append(tokens)
            all_tokens.update(tokens)

        self.vocab = sorted(list(all_tokens))
        self.vocab_size = len(self.vocab)

        # Count trigram transitions C(w_{i-2}, w_{i-1}, w_i)
        for tokens in self.tokenized_sentences:
            for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
                self.transition_counts[(w1, w2)][w3] += 1

        # Compute Conditional Probability Table P(Xt = wk | Xt-2 = wi, Xt-1 = wj)
        for (w1, w2), counts in self.transition_counts.items():
            total = sum(counts.values())
            self.cpt[(w1, w2)] = {w3: count / total for w3, count in counts.items()}

    def get_cpt(self, context: Tuple[str, str]) -> Dict[str, float]:
        """Return the probability distribution P(Xt | Xt-2 = w1, Xt-1 = w2)."""
        return self.cpt.get(context, {})

    def check_normalization(self, tolerance: float = 1e-6) -> Dict[Tuple[str, str], Tuple[float, bool]]:
        """
        Verify that for every observed context (w1, w2):
            sum_v P(v | w1, w2) == 1.0
        """
        results = {}
        for context, dist in self.cpt.items():
            total = sum(dist.values())
            is_valid = abs(total - 1.0) < tolerance
            results[context] = (total, is_valid)
        return results

    def predict_next(self, context: Tuple[str, str]) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Return (argmax_word, max_prob, distribution) for a given bigram context.
        If context is unknown, returns (None, 0.0, {}).
        """
        dist = self.get_cpt(context)
        if not dist:
            return None, 0.0, {}
        sorted_candidates = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        best_word, best_prob = sorted_candidates[0]
        return best_word, best_prob, dist

    def generate_sentence(
        self,
        mode: str = "sampling",
        max_length: int = 25,
        seed: Optional[int] = None,
        tie_break: str = "alpha",
    ) -> List[str]:
        """
        Generate a sentence using ancestral sampling or greedy argmax.
        Modes:
            - 'greedy': Always chooses argmax_w P(w | w_{t-2}, w_{t-1})
            - 'sampling': Samples w ~ P(w | w_{t-2}, w_{t-1})
        """
        rng = random.Random(seed) if seed is not None else random
        tokens = [START_TOKEN, START_TOKEN]

        while tokens[-1] != END_TOKEN and len(tokens) < max_length:
            ctx = (tokens[-2], tokens[-1])
            dist = self.get_cpt(ctx)
            if not dist:
                break

            if mode == "greedy":
                if tie_break == "alpha":
                    sorted_cand = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
                    next_word = sorted_cand[0][0]
                else:
                    max_p = max(dist.values())
                    best_words = [w for w, p in dist.items() if p == max_p]
                    next_word = best_words[0]
            elif mode == "sampling":
                words, probs = zip(*dist.items())
                next_word = rng.choices(words, weights=probs, k=1)[0]
            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            tokens.append(next_word)

        return tokens


def format_sentence(tokens: List[str], order: int = 1) -> str:
    """Format token list into clean readable sentence string."""
    start_offset = 1 if order == 1 else 2
    content_tokens = tokens[start_offset:]
    if content_tokens and content_tokens[-1] == END_TOKEN:
        content_tokens = content_tokens[:-1]
    return " ".join(content_tokens)


def compute_model_statistics(lm_1st: FirstOrderBayesianLM, lm_2nd: SecondOrderBayesianLM) -> Dict[str, Any]:
    """Compute structural and statistical metrics comparing 1st and 2nd order models."""
    V = len(lm_1st.vocab)

    # First-order parameter counts
    obs_contexts_1st = len(lm_1st.cpt)
    # Non-zero parameters (entries in transition table)
    non_zero_params_1st = sum(len(d) for d in lm_1st.cpt.values())
    # Free parameters (degrees of freedom: sum(k - 1) for each multinomial row)
    free_params_1st = sum(len(d) - 1 for d in lm_1st.cpt.values())
    # Theoretical context count: V - 1 (since <END> cannot be a conditioning prefix)
    theor_contexts_1st = V - 1
    # Theoretical entries in full transition matrix: (V - 1) * (V - 1)
    theor_entries_1st = theor_contexts_1st * (V - 1)
    zero_entries_1st = theor_entries_1st - non_zero_params_1st
    zero_entry_pct_1st = (zero_entries_1st / theor_entries_1st) * 100.0

    # Second-order parameter counts
    obs_contexts_2nd = len(lm_2nd.cpt)
    non_zero_params_2nd = sum(len(d) for d in lm_2nd.cpt.values())
    free_params_2nd = sum(len(d) - 1 for d in lm_2nd.cpt.values())
    # Theoretical contexts: V^2 (all possible bigram pairs)
    theor_contexts_2nd = V * V
    theor_entries_2nd = theor_contexts_2nd * V
    zero_contexts_2nd = theor_contexts_2nd - obs_contexts_2nd
    zero_context_pct_2nd = (zero_contexts_2nd / theor_contexts_2nd) * 100.0
    zero_entries_2nd = theor_entries_2nd - non_zero_params_2nd
    zero_entry_pct_2nd = (zero_entries_2nd / theor_entries_2nd) * 100.0

    return {
        "vocab_size": V,
        "first_order": {
            "observed_contexts": obs_contexts_1st,
            "theoretical_contexts": theor_contexts_1st,
            "non_zero_parameters": non_zero_params_1st,
            "free_parameters": free_params_1st,
            "theoretical_entries": theor_entries_1st,
            "zero_entries": zero_entries_1st,
            "zero_entry_pct": zero_entry_pct_1st,
        },
        "second_order": {
            "observed_contexts": obs_contexts_2nd,
            "theoretical_contexts": theor_contexts_2nd,
            "zero_contexts": zero_contexts_2nd,
            "zero_context_pct": zero_context_pct_2nd,
            "non_zero_parameters": non_zero_params_2nd,
            "free_parameters": free_params_2nd,
            "theoretical_entries": theor_entries_2nd,
            "zero_entries": zero_entries_2nd,
            "zero_entry_pct": zero_entry_pct_2nd,
        },
    }


def main():
    print("=" * 80)
    print(" BAYESIAN NETWORKS & AUTOREGRESSIVE LANGUAGE MODELS: EXPERIMENTAL SUITE ")
    print("=" * 80)

    # 1. Corpus and Vocabulary
    print("\n[PART III] Training Corpus:")
    for idx, sent in enumerate(DEFAULT_CORPUS, 1):
        print(f"  ({idx}) \"{sent}\"")

    lm_1st = FirstOrderBayesianLM(DEFAULT_CORPUS)
    print(f"\nVocabulary (V={lm_1st.vocab_size} tokens):")
    print(" ", lm_1st.vocab)

    # 2. Part IV: First-Order CPT Display
    print("\n" + "=" * 80)
    print("[PART IV] First-Order Conditional Probability Tables (Selected Words)")
    print("=" * 80)
    target_words = [START_TOKEN, "the", "cat", "dog", "sat", "ran"]
    for w in target_words:
        dist = lm_1st.get_cpt(w)
        counts = lm_1st.transition_counts.get(w, Counter())
        tot = sum(counts.values())
        print(f"\nP(Xt | Xt-1 = '{w}')  [Observed Count = {tot}]:")
        if not dist:
            print("  (No transitions observed)")
            continue
        for next_w, p in sorted(dist.items(), key=lambda x: -x[1]):
            c = counts[next_w]
            print(f"    -> '{next_w}': {p:.4f} (count: {c}/{tot})")
        # List zero probability transitions
        zero_transitions = [v for v in lm_1st.vocab if v not in dist]
        print(f"    [Zero-probability transitions ({len(zero_transitions)}/{lm_1st.vocab_size})]: {zero_transitions}")

    # 3. Part VII: Probability Normalisation Test
    print("\n" + "=" * 80)
    print("[PART VII] Probability Normalisation Test: sum_v P(v | w) == 1.0")
    print("=" * 80)
    norm_1st = lm_1st.check_normalization()
    all_passed_1st = True
    for w, (total, valid) in sorted(norm_1st.items()):
        status = "PASSED" if valid else "FAILED"
        if not valid:
            all_passed_1st = False
        print(f"  Context '{w:<8}': sum = {total:10.8f} -> [{status}]")
    print(f"First-Order Normalisation Check: {'ALL TESTS PASSED' if all_passed_1st else 'FAILED'}")

    # 4. Part VIII: Next-Word Prediction (Argmax)
    print("\n" + "=" * 80)
    print("[PART VIII] Next-Word Prediction & Argmax Identification")
    print("=" * 80)
    eval_words = [START_TOKEN, "the", "cat", "dog", "sat", "ran", "on", "to"]
    for w in eval_words:
        best_w, best_p, dist = lm_1st.predict_next(w)
        print(f"\nContext Xt-1 = '{w}':")
        for cand, p in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    P('{cand}' | '{w}') = {p:.4f}")
        print(f"  >> argmax_w P(w | '{w}') = '{best_w}' (p = {best_p:.4f})")

    # 5. Part IX: Generate 20 Sentences (Sampling Mode)
    print("\n" + "=" * 80)
    print("[PART IX] Generating 20 Sentences via First-Order Sampling Mode (Ancestral Sampling)")
    print("=" * 80)
    sampled_20 = []
    random.seed(101)  # Fixed seed for reproducibility
    for i in range(20):
        tokens = lm_1st.generate_sentence(mode="sampling", max_length=25)
        sent = format_sentence(tokens, order=1)
        terminated = tokens[-1] == END_TOKEN
        sampled_20.append((sent, terminated, len(tokens) - 2))
        term_str = "TERMINATED" if terminated else "CUTOFF_MAX_LEN"
        print(f"  Sentence {i+1:02d}: \"{sent}\" [{term_str}, len={len(tokens)-2}]")

    # Save to file
    out_20_path = "lab_bn/generated_sentences_20.txt"
    with open(out_20_path, "w", encoding="utf-8") as f:
        f.write("# 20 First-Order Sampled Sentences (Ancestral Sampling)\n")
        for i, (sent, term, l) in enumerate(sampled_20, 1):
            f.write(f"{i:02d}. {sent} (len={l}, terminated={term})\n")
    print(f"\n[Saved 20 sampled sentences to {out_20_path}]")

    # 6. Part X: Greedy vs Sampling Mode (5 Sentences Each)
    print("\n" + "=" * 80)
    print("[PART X] First-Order Mode A (Greedy) vs Mode B (Sampling) Comparison")
    print("=" * 80)
    print("\n--- Mode A: Greedy Generation (Always Argmax) ---")
    greedy_5 = []
    for i in range(5):
        tokens = lm_1st.generate_sentence(mode="greedy", max_length=20)
        sent = format_sentence(tokens, order=1)
        term = tokens[-1] == END_TOKEN
        greedy_5.append(sent)
        print(f"  Greedy {i+1}: \"{sent}\" [terminated={term}]")

    print("\n--- Mode B: Sampling Generation (Ancestral Sampling) ---")
    sampling_5 = []
    random.seed(202)
    for i in range(5):
        tokens = lm_1st.generate_sentence(mode="sampling", max_length=20)
        sent = format_sentence(tokens, order=1)
        term = tokens[-1] == END_TOKEN
        sampling_5.append(sent)
        print(f"  Sampling {i+1}: \"{sent}\" [terminated={term}]")

    # 7. Part XI & XII: Second-Order Bayesian Model
    print("\n" + "=" * 80)
    print("[PARTS XI & XII] Second-Order Bayesian Network: P(Xt | Xt-2, Xt-1)")
    print("=" * 80)
    lm_2nd = SecondOrderBayesianLM(DEFAULT_CORPUS)
    print(f"Second-Order Model trained. Total observed contexts: {len(lm_2nd.cpt)}")

    print("\nKey Second-Order CPT Entries:")
    sample_contexts = [
        (START_TOKEN, START_TOKEN),
        (START_TOKEN, "the"),
        ("the", "cat"),
        ("cat", "sat"),
        ("sat", "on"),
        ("on", "the"),
        ("the", "mat"),
        ("the", "rug"),
        ("the", "park"),
        ("to", "the"),
    ]
    for ctx in sample_contexts:
        dist = lm_2nd.get_cpt(ctx)
        counts = lm_2nd.transition_counts.get(ctx, Counter())
        tot = sum(counts.values())
        print(f"\nContext P(Xt | Xt-2='{ctx[0]}', Xt-1='{ctx[1]}') [Count={tot}]:")
        for next_w, p in sorted(dist.items(), key=lambda x: -x[1]):
            c = counts[next_w]
            print(f"    -> '{next_w}': {p:.4f} (count: {c}/{tot})")

    # Second-order normalisation check
    print("\nSecond-Order Normalisation Verification:")
    norm_2nd = lm_2nd.check_normalization()
    all_passed_2nd = True
    for ctx, (tot, valid) in sorted(norm_2nd.items()):
        if not valid:
            all_passed_2nd = False
        print(f"  Context {str(ctx):<25}: sum = {tot:10.8f} -> [{'PASSED' if valid else 'FAILED'}]")
    print(f"Second-Order Normalisation Check: {'ALL TESTS PASSED' if all_passed_2nd else 'FAILED'}")

    print("\n--- Second-Order Mode A (Greedy) Generation (5 sentences) ---")
    for i in range(5):
        tokens = lm_2nd.generate_sentence(mode="greedy", max_length=20)
        sent = format_sentence(tokens, order=2)
        term = tokens[-1] == END_TOKEN
        print(f"  Greedy {i+1}: \"{sent}\" [terminated={term}]")

    print("\n--- Second-Order Mode B (Sampling) Generation (10 sentences) ---")
    random.seed(303)
    second_samples = []
    for i in range(10):
        tokens = lm_2nd.generate_sentence(mode="sampling", max_length=20)
        sent = format_sentence(tokens, order=2)
        term = tokens[-1] == END_TOKEN
        second_samples.append(sent)
        print(f"  Sampling {i+1:02d}: \"{sent}\" [terminated={term}]")

    # 8. Part XIII: Detailed Model Comparison
    print("\n" + "=" * 80)
    print("[PART XIII] Quantitative & Qualitative Comparison: First-Order vs Second-Order")
    print("=" * 80)
    stats = compute_model_statistics(lm_1st, lm_2nd)

    print("\nStructural & Parameter Comparison:")
    print(f"  Vocabulary Size (|V|): {stats['vocab_size']}")
    print("  " + "-" * 70)
    print(f"  {'Metric':<35} | {'First-Order':<15} | {'Second-Order':<15}")
    print("  " + "-" * 70)
    print(f"  {'Conditioning Context':<35} | {'Xt-1 (1 token)':<15} | {'(Xt-2, Xt-1) (2)':<15}")
    print(f"  {'Observed Contexts':<35} | {stats['first_order']['observed_contexts']:<15} | {stats['second_order']['observed_contexts']:<15}")
    print(f"  {'Theoretical Contexts':<35} | {stats['first_order']['theoretical_contexts']:<15} | {stats['second_order']['theoretical_contexts']:<15}")
    print(f"  {'Zero-Probability Contexts':<35} | {stats['first_order']['theoretical_contexts'] - stats['first_order']['observed_contexts']:<15} | {stats['second_order']['zero_contexts']:<15} ({stats['second_order']['zero_context_pct']:.1f}%)")
    print(f"  {'Non-Zero Parameter Count':<35} | {stats['first_order']['non_zero_parameters']:<15} | {stats['second_order']['non_zero_parameters']:<15}")
    print(f"  {'Free Parameters (sum k-1)':<35} | {stats['first_order']['free_parameters']:<15} | {stats['second_order']['free_parameters']:<15}")
    print(f"  {'Theoretical Parameter Space':<35} | {stats['first_order']['theoretical_entries']:<15} | {stats['second_order']['theoretical_entries']:<15}")
    print(f"  {'Zero Entries in Full Table':<35} | {stats['first_order']['zero_entries']:<15} | {stats['second_order']['zero_entries']:<15} ({stats['second_order']['zero_entry_pct']:.1f}%)")
    print("  " + "-" * 70)

    # Diversity comparison across 100 generations
    random.seed(404)
    samples_1st_100 = [format_sentence(lm_1st.generate_sentence(mode="sampling", max_length=25), order=1) for _ in range(100)]
    random.seed(404)
    samples_2nd_100 = [format_sentence(lm_2nd.generate_sentence(mode="sampling", max_length=25), order=2) for _ in range(100)]

    unique_1st = len(set(samples_1st_100))
    unique_2nd = len(set(samples_2nd_100))
    print(f"\nEmpirical Generation Diversity (100 sampled runs):")
    print(f"  First-Order Unique Sentences : {unique_1st} / 100")
    print(f"  Second-Order Unique Sentences: {unique_2nd} / 100 (Total possible valid sentences = 6)")

    print("\n" + "=" * 80)
    print(" EXECUTION AND VERIFICATION COMPLETE ")
    print("=" * 80)


if __name__ == "__main__":
    main()
