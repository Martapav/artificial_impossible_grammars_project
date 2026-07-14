"""Builds a HuggingFace-compatible tokenizer over the generated lexicon.

Morphemes (stems and inflectional digits) are the atomic tokens; a word with
inflection is a `#`-joined sequence of morphemes (e.g. "n#4#2#7"). Vocabulary
is read directly off lexicon.json rather than trained from corpus text, since
the lexicon is the closed, authoritative set of surface forms.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

LEXICON_PATH = Path(__file__).resolve().parent.parent / "lexicon.json"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

LEXICAL_CATEGORIES = (
    "cat1", "cat1pron", "cat2", "cat3", "cat3aux",
    "cat4", "cat5", "cat6", "cat8", "cat9",
)


def load_lexicon(path: Path = LEXICON_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_morphemes(lexicon: dict) -> set[str]:
    """All surface morphemes: lexical stems plus inflectional digits.

    Excludes the null morpheme (config_snapshot["NULL_MORPHEME"]): it has no
    surface form.
    """
    morphemes: set[str] = set()
    for category in LEXICAL_CATEGORIES:
        morphemes.update(item["form"] for item in lexicon[category])
    for parameter in lexicon["inflectional"]["parameters"]:
        morphemes.update(parameter["values"])

    null_morpheme = lexicon["metadata"]["config_snapshot"]["NULL_MORPHEME"]
    morphemes.discard(null_morpheme)
    return morphemes


def build_tokenizer(lexicon_path: Path = LEXICON_PATH) -> PreTrainedTokenizerFast:
    lexicon = load_lexicon(lexicon_path)
    config = lexicon["metadata"]["config_snapshot"]
    separator = config["SEPARATOR"]
    null_morpheme = config["NULL_MORPHEME"]

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    for morpheme in sorted(collect_morphemes(lexicon)):
        vocab[morpheme] = len(vocab)

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Replace(null_morpheme, "")
    # Sentences separate words with whitespace and morphemes with SEPARATOR;
    # both boundaries must be split so each token is exactly one morpheme.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(separator, behavior="removed"),
    ])
    # Every sentence is an independent sequence: mark its boundaries so the
    # model can condition the first morpheme and learn where sentences end.
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, vocab[BOS_TOKEN]), (EOS_TOKEN, vocab[EOS_TOKEN])],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
    )


tokenizer = build_tokenizer()
