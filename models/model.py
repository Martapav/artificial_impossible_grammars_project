"""Instantiates the GPT-2 causal LM to be trained on the artificial grammars.

"""

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed

from models.tokenize import tokenizer as default_tokenizer

# Default architecture point (grid search sweeps around this) 
# Ranges the grid search is expected to explore: n_layer 2-6, n_embd 128-256.
DEFAULT_N_LAYER = 4
DEFAULT_N_EMBD = 256
DEFAULT_N_HEAD = 8
DEFAULT_N_POSITIONS = 256
DEFAULT_DROPOUT = 0.1

# Longest tokenized sequence across all corpora/splits (BOS/EOS included),
# driven by the test_generalization split. Recompute after regenerating
# corpora: `python -m models.dataload` reports per-split maxima.
OBSERVED_MAX_SEQ_LEN = 108


def build_config(
    *,
    n_layer: int = DEFAULT_N_LAYER,
    n_embd: int = DEFAULT_N_EMBD,
    n_head: int = DEFAULT_N_HEAD,
    n_positions: int = DEFAULT_N_POSITIONS,
    dropout: float = DEFAULT_DROPOUT,
    vocab_size: int | None = None,
    tokenizer: PreTrainedTokenizerFast = default_tokenizer,
) -> GPT2Config:
    """Builds a validated GPT2Config for our vocabulary.

    `vocab_size` defaults to `len(tokenizer)` (the closed morpheme vocabulary).
    Special-token ids (bos/eos/pad) are copied from the tokenizer so that
    generation, evaluation, and the padded-loss mask all agree with training.
    """
    if vocab_size is None:
        vocab_size = len(tokenizer)

    if n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head}); "
            f"head dim would be {n_embd / n_head}."
        )
    if n_positions < OBSERVED_MAX_SEQ_LEN:
        raise ValueError(
            f"n_positions ({n_positions}) is below the longest tokenized "
            f"sequence ({OBSERVED_MAX_SEQ_LEN}); those positions would overflow "
            f"the positional-embedding table at eval time."
        )

    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        # Single knob for the three GPT-2 dropout sites.
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        # Keep the model's notion of the special tokens aligned with the
        # tokenizer that produced the training ids.
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def build_model(
    *,
    n_layer: int = DEFAULT_N_LAYER,
    n_embd: int = DEFAULT_N_EMBD,
    n_head: int = DEFAULT_N_HEAD,
    n_positions: int = DEFAULT_N_POSITIONS,
    dropout: float = DEFAULT_DROPOUT,
    vocab_size: int | None = None,
    tokenizer: PreTrainedTokenizerFast = default_tokenizer,
    seed: int | None = None,
) -> GPT2LMHeadModel:
    """Instantiates a fresh (untrained) GPT2LMHeadModel from a fresh config.

    Pass `seed` to make weight initialisation reproducible: it is applied
    immediately before `from_config`, which is where GPT-2 initialises its
    parameters.
    """
    config = build_config(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        dropout=dropout,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
    )

    if seed is not None:
        set_seed(seed)

    # `from_config` is an AutoModel classmethod; the concrete-class equivalent
    # is direct construction, which initialises fresh (untrained) weights.
    return GPT2LMHeadModel(config)


def count_parameters(model: GPT2LMHeadModel) -> tuple[int, int]:
    """(total, trainable) parameter counts.

    Input and output embeddings are tied in GPT2LMHeadModel, so the shared
    weight is counted once.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    """Instantiate the default model and print a one-line summary sanity check."""
    model = build_model(seed=0)
    config = model.config
    total, trainable = count_parameters(model)
    print(
        f"GPT2LMHeadModel  "
        f"layers={config.n_layer} n_embd={config.n_embd} heads={config.n_head} "
        f"head_dim={config.n_embd // config.n_head}  "
        f"n_positions={config.n_positions} vocab={config.vocab_size}  "
        f"bos={config.bos_token_id} eos={config.eos_token_id} pad={config.pad_token_id}"
    )
    print(f"parameters: total={total:,}  trainable={trainable:,}")


if __name__ == "__main__":
    main()
