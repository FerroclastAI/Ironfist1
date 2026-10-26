
"""
IronFist1 Model Configuration
=============================

General-purpose language model:
- Framework: PyTorch
- Tokenizer: WordPiece
- Training: From scratch
- Reasoning: rStar-inspired MCTS
- Architecture: Policy model + discriminator model

Parameter counts are targets, not guarantees. The actual counts
depend on the implemented architecture and vocabulary size.
"""

from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    """WordPiece tokenizer configuration."""

    vocab_size: int = 32_000
    min_frequency: int = 2
    lowercase: bool = False
    max_input_chars_per_word: int = 100

    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"

    @property
    def special_tokens(self) -> list[str]:
        return [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
        ]


@dataclass
class TransformerConfig:
    """Shared Transformer architecture settings."""

    vocab_size: int = 32_000
    max_seq_len: int = 2048

    hidden_size: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    tie_word_embeddings: bool = True
    use_bias: bool = False

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")


@dataclass
class PolicyConfig(TransformerConfig):
    """Autoregressive model that generates candidate solutions."""

    target_parameters: int = 500_000_000

    # Training objective: predict the next token.
    objective: str = "causal_language_modeling"

    # Longer sequences can be enabled after initial training works.
    use_gradient_checkpointing: bool = True


@dataclass
class DiscriminatorConfig(TransformerConfig):
    """Model that evaluates candidate reasoning steps."""

    target_parameters: int = 150_000_000

    # Predict a quality score for a candidate solution.
    objective: str = "reasoning_quality"

    # Score the complete candidate using a classification/regression head.
    num_score_classes: int = 1
    use_gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Shared training settings."""

    seed: int = 42

    # Mixed precision and memory management.
    precision: str = "bf16"
    gradient_checkpointing: bool = True

    # Optimizer.
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Training schedule.
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_steps: int = 100_000
    warmup_steps: int = 2_000

    # Data loading.
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpoints.
    save_every_steps: int = 1_000
    log_every_steps: int = 10
    output_dir: str = "checkpoints"


@dataclass
class RStarConfig:
    """Mutual reasoning and Monte Carlo tree search settings."""

    enabled: bool = True

    # Search budget.
    num_simulations: int = 32
    max_search_depth: int = 8
    max_candidates_per_node: int = 4

    # MCTS exploration.
    exploration_constant: float = 1.414
    discount_factor: float = 0.99

    # Candidate generation.
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512

    # Policy/discriminator interaction.
    discriminator_weight: float = 1.0
    consistency_weight: float = 0.5


@dataclass
class IronFistConfig:
    """Top-level configuration for IronFist1."""

    name: str = "IronFist1"
    version: str = "0.1.0"

    tokenizer: TokenizerConfig | None = None
    policy: PolicyConfig | None = None
    discriminator: DiscriminatorConfig | None = None
    training: TrainingConfig | None = None
    rstar: RStarConfig | None = None

    def __post_init__(self) -> None:
        self.tokenizer = self.tokenizer or TokenizerConfig()

        self.policy = self.policy or PolicyConfig(
            vocab_size=self.tokenizer.vocab_size
        )

        self.discriminator = self.discriminator or DiscriminatorConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            max_seq_len=self.policy.max_seq_len,
        )

        self.training = self.training or TrainingConfig()
        self.rstar = self.rstar or RStarConfig()

        self.validate()

    def validate(self) -> None:
        self.policy.validate()
        self.discriminator.validate()

        if self.policy.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "Policy vocabulary must match the tokenizer vocabulary."
            )

        if self.discriminator.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "Discriminator vocabulary must match the tokenizer vocabulary."
            )

        if self.policy.max_seq_len != self.discriminator.max_seq_len:
            raise ValueError(
                "Policy and discriminator must use the same context length."
            )

        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )

        if self.rstar.num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")

        if self.rstar.max_search_depth <= 0:
            raise ValueError("max_search_depth must be positive.")


# Default project configuration.
CONFIG = IronFistConfig()
