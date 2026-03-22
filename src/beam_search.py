"""
Beam Search Decoding
====================

Beam search is a decoding strategy that explores multiple prediction paths
simultaneously, keeping only the top-scoring candidates at each step.
It's widely used in machine translation, speech recognition, and text generation.

EDUCATIONAL CONTEXT:
-------------------
Without beam search, language models typically use "greedy decoding" — at each
step, pick the single most likely next token. This is fast but often suboptimal
because a locally-best choice can lead to a globally-bad sequence.

BEAM SEARCH solves this by maintaining BEAM_WIDTH parallel hypotheses and
expanding all of them at each step, then pruning to keep only the best ones.

ANALOGY:
Imagine you're navigating a maze:
- Greedy: Always take the corridor that looks most promising. Might get stuck.
- Beam search: Send BEAM_WIDTH explorers down different corridors. At each
  junction, keep only the best explorers. More likely to find the best path.

ALGORITHM:
----------
1. Start with beam_width copies of the initial context
2. For each step:
   a. For each hypothesis in the beam, get top-k next token predictions
   b. Compute score = log_probability_sum / sequence_length^alpha
   c. Keep only the top beam_width hypotheses (prune the rest)
3. After max_steps or when all beams hit an end token, return the best sequence

LENGTH PENALTY (alpha):
- alpha < 1.0: Favors longer sequences (prevents premature stopping)
- alpha = 1.0: Standard length normalization
- alpha > 1.0: Favors shorter sequences (more concise)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BeamSearchDecoder:
    """Beam search decoder for text autocomplete.

    This class wraps any language model that provides predict_next() method
    and enhances it with beam search for better multi-step predictions.

    Instead of greedily picking the most likely next word at each step,
    beam search explores multiple candidate sequences in parallel and
    returns the sequence with the highest overall probability.

    Attributes:
        beam_width: Number of parallel hypotheses to maintain.
        max_length: Maximum number of tokens to generate beyond the input.
        length_penalty: Exponent for length normalization (see class docstring).
    """

    def __init__(
        self,
        beam_width: int = 5,
        max_length: int = 10,
        length_penalty: float = 0.6,
    ) -> None:
        """Initialize the beam search decoder.

        Args:
            beam_width: Number of candidate sequences to keep at each step.
                Higher = more thorough search but slower. 1 = greedy decoding.
                Typical values: 3-10. Beyond 10 rarely helps.
            max_length: Maximum number of NEW tokens to generate.
                Total sequence length = len(input) + max_length.
            length_penalty: Length normalization exponent.
                0.6 is a common default that slightly favors longer sequences.
                This prevents the decoder from stopping too early (shorter
                sequences often have higher average log-probability simply
                because there are fewer terms to average over).
        """
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty

    def _length_normalized_score(self, log_prob: float, length: int) -> float:
        """Compute length-normalized log probability score.

        WHY LENGTH NORMALIALIZATION?
        Without it, beam search strongly prefers shorter sequences because:
        - Each additional token multiplies by a probability < 1.0
        - So log_prob gets more negative with each step
        - Short sequences end up with higher (less negative) scores

        The normalization divides by length^alpha:
            score = log_prob / length^alpha

        This levels the playing field between short and long sequences.

        Args:
            log_prob: Sum of log probabilities for the sequence.
            length: Number of tokens in the sequence (beyond input).

        Returns:
            Normalized score (higher is better).
        """
        if length == 0:
            return log_prob
        return log_prob / (length ** self.length_penalty)

    def search(
        self,
        model: Any,
        context_tokens: List[str],
        steps: Optional[int] = None,
        candidates_per_step: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run beam search to find the best continuation sequences.

        THE BEAM SEARCH LOOP:
        Think of it as a tournament bracket:
        1. Start: [initial_context] (1 hypothesis)
        2. Expand: For each hypothesis, generate candidates_per_step next words
        3. Score: Compute the normalized score for each expanded hypothesis
        4. Prune: Keep only the top beam_width hypotheses
        5. Repeat from step 2 for max_length steps

        Args:
            model: Any model with a predict_next(context, top_k) method
                that returns List[Tuple[str, float]] of (token, probability).
            context_tokens: The input context tokens.
            steps: Number of generation steps (defaults to self.max_length).
            candidates_per_step: How many next-token candidates to consider
                at each expansion step per beam. Should be >= beam_width.

        Returns:
            List of beam results, sorted by score (best first). Each result is a dict:
            {
                "tokens": [word1, word2, ...],      # Full generated sequence
                "score": float,                       # Length-normalized log prob
                "log_prob": float,                    # Raw sum of log probs
                "length": int,                        # Number of generated tokens
            }
        """
        if steps is None:
            steps = self.max_length

        # Each beam is: (list_of_generated_tokens, log_probability)
        # We start with one empty beam (no tokens generated yet)
        beams: List[Tuple[List[str], float]] = [([], 0.0)]

        for step in range(steps):
            all_candidates: List[Tuple[List[str], float]] = []

            # EXPAND: for each active beam, get next-token predictions
            for tokens, log_prob in beams:
                # Build the full context: original input + tokens generated so far
                full_context = context_tokens + tokens

                # Get predictions from the model
                predictions = model.predict_next(full_context, top_k=candidates_per_step)

                for word, prob in predictions:
                    if prob <= 0:
                        continue  # Skip zero-probability tokens

                    # Extend this beam with the predicted word
                    new_tokens = tokens + [word]
                    # Add log(prob) to cumulative log probability
                    # (we use log because multiplying many small probabilities
                    # causes numerical underflow — log turns multiplication
                    # into addition, which is numerically stable)
                    new_log_prob = log_prob + np.log(prob)
                    all_candidates.append((new_tokens, new_log_prob))

            if not all_candidates:
                logger.warning("Beam search: no candidates at step %d, stopping early", step)
                break

            # SCORE & PRUNE: keep only the top beam_width candidates
            # We use length-normalized scoring to fairly compare sequences
            # of different lengths
            scored = []
            for tokens, log_prob in all_candidates:
                length = len(tokens)
                norm_score = self._length_normalized_score(log_prob, length)
                scored.append((tokens, log_prob, length, norm_score))

            # Sort by normalized score (descending — higher is better)
            scored.sort(key=lambda x: x[3], reverse=True)

            # Keep only the top beam_width beams
            beams = [(tokens, log_prob) for tokens, log_prob, _, _ in scored[:self.beam_width]]

            logger.debug(
                "Beam step %d/%d: %d candidates, best score=%.4f",
                step + 1, steps, len(all_candidates), scored[0][3],
            )

        # Convert beams to result dicts, sorted by score
        results = []
        for tokens, log_prob in beams:
            length = len(tokens)
            norm_score = self._length_normalized_score(log_prob, length)
            results.append({
                "tokens": tokens,
                "score": round(norm_score, 6),
                "log_prob": round(log_prob, 6),
                "length": length,
            })

        # Best beam first
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
