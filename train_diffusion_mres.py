# MRes D1 latent-diffusion trainer.
#
# Scientific design (implementation plan sections 5 and 10):
#   released B1 TransformerDenoiser at configs/rnagenesis/diffusion is trainable;
#   the complete released EncDec at configs/rnagenesis/autoencoder stays frozen.
# B0 at checkpoints/Aptamer/diffusion is not an initialisation.
#
# This file reuses the upstream denoising step from train_diffusion.py at
# 6bc641acfb16aa2dfcbfd33b50fe5085150777a9: DDIMScheduler.add_noise, epsilon
# prediction, MSE, AdamW, cosine schedule, gradient accumulation, gradient
# clipping, Accelerator, and DDIMPipeline1D checkpoint layout.
#
# It does not reuse the upstream 90/10 split, T->U rewriting, RNA-FM loader,
# FP16 cast for every mixed-precision mode, fixed 50-step warmup, or the
# absence of a VALIDATION selection path.
#
# File newlines are record separators. Removing a trailing \\n or \\r\\n is
# allowed. Changing biological sequence characters is not: no strip, no
# upper, no T->U, no ambiguity-code conversion.

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Released EncDec imports `esm` from this directory. generation.py adds the
# same path relative to the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "models" / "autoencoder" / "encoder"))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler

from models.autoencoder.encdec import EncDec
from models.autoencoder.encoder.biomap.rna_tokens import encode_rna_characters, load_tokenizer
from models.diffusion_models.pipeline_ddim import DDIMPipeline1D
from models.diffusion_models.transformer import TransformerDenoiser


REPO_ROOT = Path(__file__).resolve().parent
B1_DIFFUSION_DIR = REPO_ROOT / "configs" / "rnagenesis" / "diffusion"
B0_DIFFUSION_DIR = REPO_ROOT / "checkpoints" / "Aptamer" / "diffusion"
ENCDEC_DIR = REPO_ROOT / "configs" / "rnagenesis" / "autoencoder"

# Measured from B1 safetensors metadata. Runtime code still counts parameters.
EXPECTED_DENOISER_PARAMETER_COUNT = 1_961_920_992

LATENT_QUERIES = 32
LATENT_CHANNELS = 160
CANONICAL_PATTERN_DESCRIPTION = "^[ACGU]+$"

# Plan section 11.1. These match the upstream AdamW defaults and are adopted
# explicitly, including gradient_clip_norm = 1.0. They are not an unread copy
# of an upstream hard-coded clip.
ADAM_BETA1 = 0.95
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1e-6
ADAM_EPSILON = 1e-8
GRADIENT_CLIP_NORM = 1.0

# Plan section 11.2. Upstream example LR is 1e-4; D1 starts at 3e-5.
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EPOCHS = 5
MAX_EPOCHS = 5
EFFECTIVE_BATCH_SIZE = 32
# Plan section 11.3. Single GPU. Effective batch is microbatch * accumulation.
ALLOWED_MICROBATCH_ACCUMULATION = {8: 4, 16: 2, 32: 1}

# Plan section 11.9 names a smoke seed but does not assign the integer.
# 42 is the upstream trainer default, used only as the implementation/smoke
# seed. Pilot and replicate seeds are supplied with --seed.
DEFAULT_TRAIN_SEED = 42
# Plan section 9.3 requires a fixed validation-evaluation seed and does not
# assign the integer or the draw count. These defaults are independent of
# --seed. One draw per sequence is the training objective's single (t, epsilon).
DEFAULT_VALIDATION_SEED = 1
DEFAULT_VALIDATION_DRAWS = 1

# Frozen contract hashes. REF and EVAL are rejection metadata only.
TRAIN_TEXT_SHA256 = "d6cbc41a36b4a83e393e5593598a2bda246bc6e9df151d236381885734e05c9d"
VALIDATION_CSV_SHA256 = "cb347a739455924a7f4c55206bd2ac0131b7f36d79c8881e37e42728b600e0cd"
FORBIDDEN_SHA256 = {
    "531dd98ff5887e349b2f0e16db463c3b60b51f5f02f912cf099a1a5eb2debf23": "REF",
    "ff4b36535052981c240dfa0571acd9cf00a386c80d9487a5994d1597f3386015": "EVAL",
}
FORBIDDEN_BASENAMES = {"ref.csv", "eval.csv"}

# Redundant with the frozen file hashes. Checked only when those hashes are enforced.
FROZEN_TRAIN_SEQUENCE_COUNT = 669
FROZEN_VALIDATION_CORE_COUNT = 95

# Required M0 Diffusers patch. Resolved from the imported module, not from an environment path.
DIFFUSERS_NORMALIZATION_SHA256 = "09bc632069f1d6e1dc1b1cb7f5b9ca096c83e3010835f11b8dc6c58b45aa13a5"

EXTERNAL_CONTRACT_VERIFIER = (
    "python3 scripts/modelling/verify_modelling_contract.py "
    "--profile development --data-dir /home/laura/rnagenesis-mres-data/development"
)


@dataclass(frozen=True)
class UpdatePlan:
    train_examples: int
    microbatch: int
    gradient_accumulation: int
    effective_batch: int
    dataloader_batches_per_epoch: int
    optimizer_updates_per_epoch: int
    epochs: int
    total_optimizer_updates: int
    warmup_updates: int


@dataclass(frozen=True)
class ValidationDraws:
    """Fixed timesteps and noise, one row per validation sequence then draw."""

    seed: int
    draws_per_sequence: int
    num_train_timesteps: int
    sequences_sha256: str
    timesteps: torch.Tensor  # [N, K] int64
    noise: torch.Tensor  # [N, K, 32, 160]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diffusers_normalization_source_path() -> Path:
    """Source file of the diffusers.models.normalization module this process imported."""
    import diffusers.models.normalization as normalization

    source = inspect.getsourcefile(normalization)
    if not source:
        raise RuntimeError("cannot resolve the source file of diffusers.models.normalization")
    return Path(source)


def assert_diffusers_normalization_patch(path: Optional[Path] = None) -> str:
    """Fail closed unless the imported Diffusers normalization source matches the M0 patch."""
    source = Path(path) if path is not None else diffusers_normalization_source_path()
    digest = sha256_file(source)
    if digest != DIFFUSERS_NORMALIZATION_SHA256:
        raise RuntimeError(
            "diffusers normalization source %s has SHA256 %s; required patch is %s"
            % (source, digest, DIFFUSERS_NORMALIZATION_SHA256)
        )
    return digest


def assert_frozen_role_counts(train_sequences: int, validation_cores: int) -> None:
    """Checked only for the frozen corpus. Fixture runs do not call this."""
    if train_sequences != FROZEN_TRAIN_SEQUENCE_COUNT:
        raise ValueError(
            "frozen TRAIN produced %d sequences; expected %d"
            % (train_sequences, FROZEN_TRAIN_SEQUENCE_COUNT)
        )
    if validation_cores != FROZEN_VALIDATION_CORE_COUNT:
        raise ValueError(
            "frozen VALIDATION produced %d unique sequence cores; expected %d"
            % (validation_cores, FROZEN_VALIDATION_CORE_COUNT)
        )


def assert_output_isolated(output: Path, input_paths: Sequence[Path]) -> Path:
    """Reject an output that is an input file, that file's parent, or anything beneath that parent.

    Comparison uses resolved paths only. It does not open the inputs, glob, or create the output.
    """
    if not output.is_absolute():
        output = Path.cwd() / output
    output_resolved = output.resolve()
    for input_path in input_paths:
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        resolved_input = input_path.resolve()
        parent = resolved_input.parent
        under_parent = output_resolved == parent or parent in output_resolved.parents
        if output_resolved == resolved_input or under_parent:
            raise ValueError(
                "refusing output %s: it is an input file or lies in or under an input parent %s"
                % (output_resolved, parent)
            )
    return output_resolved


def split_sequence_records(text: str) -> List[str]:
    """Split a sequence file on newline delimiters only.

    A trailing newline is the record terminator and does not create an empty
    sequence. A carriage return immediately before that newline is the other
    half of a CRLF delimiter. Spaces and sequence characters are kept so a
    malformed line fails the alphabet check instead of being repaired.
    """
    if text.startswith("\ufeff"):
        raise ValueError("sequence file begins with a BOM; refusing to rewrite it")
    if text.endswith("\n"):
        text = text[:-1]
    if text.endswith("\r"):
        text = text[:-1]
    if text == "":
        raise ValueError("sequence file contains no records")
    records = []
    for record in text.split("\n"):
        if record.endswith("\r"):
            record = record[:-1]
        records.append(record)
    return records


def assert_canonical_sequence(sequence: str, location: str) -> None:
    if not isinstance(sequence, str) or any(character not in "ACGU" for character in sequence) or sequence == "":
        raise ValueError(
            "%s is not a nonempty canonical ACGU sequence (%s): %r"
            % (location, CANONICAL_PATTERN_DESCRIPTION, sequence)
        )


def assert_allowed_input_path(path: Path) -> Path:
    """Reject REF/EVAL names before the file is opened. Do not search directories."""
    if not path.is_absolute():
        path = (Path.cwd() / path)
    if path.is_symlink():
        raise ValueError("refusing symlink input: %s" % path)
    if any(part in FORBIDDEN_BASENAMES for part in path.parts):
        raise ValueError("REF/EVAL paths cannot be supplied to the D1 trainer: %s" % path)
    if not path.is_file():
        raise ValueError("input path is not a regular file: %s" % path)
    return path


def assert_hashes_allowed(digest: str, role: str) -> None:
    forbidden = FORBIDDEN_SHA256.get(digest)
    if forbidden is not None:
        raise ValueError("%s file matches the frozen %s hash; refusing to read it as trainer input" % (role, forbidden))


def read_train_sequences(path: Path, enforce_frozen_hash: bool) -> List[str]:
    path = assert_allowed_input_path(path)
    digest = sha256_file(path)
    assert_hashes_allowed(digest, "TRAIN")
    if enforce_frozen_hash and digest != TRAIN_TEXT_SHA256:
        raise ValueError(
            "TRAIN file hash %s does not match frozen corpus_final.txt %s" % (digest, TRAIN_TEXT_SHA256)
        )
    text = path.read_bytes().decode("utf-8")
    sequences = split_sequence_records(text)
    seen = set()
    for index, sequence in enumerate(sequences, start=1):
        assert_canonical_sequence(sequence, "TRAIN line %d" % index)
        if sequence in seen:
            raise ValueError("TRAIN line %d repeats an earlier sequence; refusing to drop it" % index)
        seen.add(sequence)
    return sequences


def read_validation_cores(path: Path, enforce_frozen_hash: bool) -> List[str]:
    """First-occurrence unique sequence_core values from ft_validation.csv.

    The frozen contract's validation grain is one distinct VALIDATION core.
    Identity rows that repeat a core are collapsed in file order. This is the
    same first-occurrence rule that built corpus_final.txt from TRAIN. It is
    not a new split and it is not applied to the TRAIN text file.
    """
    path = assert_allowed_input_path(path)
    digest = sha256_file(path)
    assert_hashes_allowed(digest, "VALIDATION")
    if enforce_frozen_hash and digest != VALIDATION_CSV_SHA256:
        raise ValueError(
            "VALIDATION file hash %s does not match frozen ft_validation.csv %s"
            % (digest, VALIDATION_CSV_SHA256)
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sequence_core" not in reader.fieldnames:
            raise ValueError("VALIDATION CSV must contain a sequence_core column")
        if enforce_frozen_hash and "ft_split" not in reader.fieldnames:
            raise ValueError("VALIDATION CSV must contain ft_split")
        cores: List[str] = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            if enforce_frozen_hash and row.get("ft_split") != "VALIDATION":
                raise ValueError("VALIDATION row %d ft_split is not VALIDATION" % row_number)
            core = row.get("sequence_core")
            if core is None:
                raise ValueError("VALIDATION row %d is missing sequence_core" % row_number)
            assert_canonical_sequence(core, "VALIDATION row %d sequence_core" % row_number)
            if core in seen:
                continue
            seen.add(core)
            cores.append(core)
    if not cores:
        raise ValueError("VALIDATION file produced no sequence cores")
    return cores


def assert_distinct_roles(train_path: Path, validation_path: Path) -> None:
    if assert_allowed_input_path(train_path).resolve() == assert_allowed_input_path(validation_path).resolve():
        raise ValueError("TRAIN and VALIDATION must be explicit different files")


def diffusion_compute_dtype(mixed_precision: str) -> torch.dtype:
    """Accelerate autocast dtype inside the prepared denoiser.

    Noise, DDIM add_noise, the epsilon target and the MSE stay float32.
    bf16 stays bfloat16. It is not mapped to float16. "no" means autocast is off.
    """
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError("mixed_precision must be no, fp16 or bf16, got %r" % (mixed_precision,))


def assert_effective_batch(microbatch: int, accumulation: int) -> None:
    expected = ALLOWED_MICROBATCH_ACCUMULATION.get(microbatch)
    if expected is None or expected != accumulation or microbatch * accumulation != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "D1 allows only microbatch/accumulation pairs %s"
            % ({k: v for k, v in ALLOWED_MICROBATCH_ACCUMULATION.items()},)
        )


def dataloader_batch_count(n_examples: int, microbatch: int) -> int:
    if n_examples < 1 or microbatch < 1:
        raise ValueError("example and microbatch counts must be positive")
    # drop_last=False: the final shorter batch is retained.
    return math.ceil(n_examples / microbatch)


def accumulation_windows(batch_sizes: Sequence[int], accumulation: int) -> List[List[int]]:
    """Split microbatch sizes into optimiser-update windows.

    A window closes after `accumulation` microbatches. A trailing window with
    fewer microbatches is kept.
    """
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    if not batch_sizes or any((not isinstance(size, int)) or isinstance(size, bool) or size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must be positive integers")
    windows: List[List[int]] = []
    start = 0
    count = len(batch_sizes)
    while start < count:
        stop = min(start + accumulation, count)
        windows.append(list(batch_sizes[start:stop]))
        start = stop
    return windows


def accumulation_window_weights(batch_sizes: Sequence[int]) -> List[float]:
    """Scale for each mean-reduced microbatch loss inside one optimiser update.

    A microbatch of B examples in a window of W examples contributes loss * B / W,
    so every example in the window has equal weight.
    """
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("window batch sizes must be positive")
    window = float(sum(batch_sizes))
    return [size / window for size in batch_sizes]


def microbatch_loss_weights(batch_sizes: Sequence[int], accumulation: int) -> List[float]:
    """Per-microbatch B/W weights, including a short final accumulation window."""
    weights: List[float] = []
    for window in accumulation_windows(batch_sizes, accumulation):
        weights.extend(accumulation_window_weights(window))
    return weights


def accelerator_backward_scale(example_weight: float, gradient_accumulation: int) -> float:
    """Loss multiplier to pass into Accelerator.backward.

    Accelerate 0.25.0 divides that loss by gradient_accumulation_steps again
    (Accelerator.backward, non-DeepSpeed). Multiplying by the accumulation count
    first leaves the net gradient scale at example_weight, which is B/W.
    """
    if gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    if example_weight <= 0:
        raise ValueError("example weight must be positive")
    return float(example_weight) * gradient_accumulation


def accumulation_step_indices(n_microbatches: int, accumulation: int) -> List[int]:
    """Microbatch indexes (0-based) that end an optimiser update.

    The final incomplete accumulation window is included. It is not dropped.
    """
    if n_microbatches < 1 or accumulation < 1:
        raise ValueError("microbatch and accumulation counts must be positive")
    indexes = []
    for index in range(n_microbatches):
        window_is_full = (index + 1) % accumulation == 0
        window_is_final = index == n_microbatches - 1
        if window_is_full or window_is_final:
            indexes.append(index)
    return indexes


def optimizer_updates_per_epoch(n_examples: int, microbatch: int, accumulation: int) -> int:
    batches = dataloader_batch_count(n_examples, microbatch)
    return len(accumulation_step_indices(batches, accumulation))


def warmup_update_count(planned_optimizer_updates: int) -> int:
    """Plan section 11.5: max(1, round(0.10 * planned optimizer updates)).

    Division by 10 keeps the halfway case exact. Python round uses
    half-to-even, so 105 updates give round(10.5) = 10.
    """
    if planned_optimizer_updates < 1:
        raise ValueError("planned optimizer updates must be positive")
    return max(1, round(planned_optimizer_updates / 10))


def build_update_plan(n_examples: int, microbatch: int, accumulation: int, epochs: int) -> UpdatePlan:
    assert_effective_batch(microbatch, accumulation)
    if epochs < 1 or epochs > MAX_EPOCHS:
        raise ValueError("D1 development runs use 1 to %d epochs" % MAX_EPOCHS)
    batches = dataloader_batch_count(n_examples, microbatch)
    updates = optimizer_updates_per_epoch(n_examples, microbatch, accumulation)
    total = updates * epochs
    return UpdatePlan(
        train_examples=n_examples,
        microbatch=microbatch,
        gradient_accumulation=accumulation,
        effective_batch=microbatch * accumulation,
        dataloader_batches_per_epoch=batches,
        optimizer_updates_per_epoch=updates,
        epochs=epochs,
        total_optimizer_updates=total,
        warmup_updates=warmup_update_count(total),
    )


def parameter_report(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def assert_complete_trainable(module: torch.nn.Module, expected_total: Optional[int] = None) -> Dict[str, int]:
    report = parameter_report(module)
    if report["trainable"] != report["total"]:
        raise AssertionError(
            "trainable parameter count %d != complete parameter count %d"
            % (report["trainable"], report["total"])
        )
    if expected_total is not None and report["total"] != expected_total:
        raise AssertionError(
            "parameter count %d != expected %d" % (report["total"], expected_total)
        )
    return report


def assert_encdec_frozen(encdec: torch.nn.Module) -> None:
    if encdec.training:
        raise AssertionError("EncDec.training is True after freeze")
    for name, parameter in encdec.named_parameters():
        if parameter.requires_grad:
            raise AssertionError("EncDec parameter %s requires grad" % name)


def freeze_encdec(encdec: EncDec) -> EncDec:
    encdec.freeze()
    assert_encdec_frozen(encdec)
    return encdec


def assert_optimizer_membership(optimizer: torch.optim.Optimizer, denoiser: torch.nn.Module, encdec: torch.nn.Module) -> None:
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in denoiser.parameters() if parameter.requires_grad}
    encdec_ids = {id(parameter) for parameter in encdec.parameters()}
    frozen_ids = {
        id(parameter)
        for parameter in list(denoiser.parameters()) + list(encdec.parameters())
        if not parameter.requires_grad
    }
    if optimizer_ids != trainable_ids:
        raise AssertionError("optimizer parameters are not exactly the trainable denoiser parameters")
    if optimizer_ids & encdec_ids:
        raise AssertionError("optimizer contains an EncDec parameter")
    if optimizer_ids & frozen_ids:
        raise AssertionError("optimizer contains a frozen parameter")


def assert_b1_initialization_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == B0_DIFFUSION_DIR.resolve():
        raise ValueError("refusing B0 initialisation at checkpoints/Aptamer/diffusion")
    if resolved != B1_DIFFUSION_DIR.resolve():
        raise ValueError("D1 must initialise from configs/rnagenesis/diffusion, got %s" % resolved)
    return resolved


def load_scheduler(diffusion_dir: Path) -> DDIMScheduler:
    scheduler = DDIMScheduler.from_pretrained(str(diffusion_dir), subfolder="scheduler")
    if scheduler.config.prediction_type != "epsilon":
        raise AssertionError("released scheduler prediction_type is not epsilon")
    return scheduler


def load_frozen_encdec(path: Path) -> EncDec:
    encdec = EncDec.from_pretrained(str(path))
    if encdec.data_type != "rna" or encdec.rna_encoder_type != "biomap":
        raise AssertionError("EncDec is not the released biomap RNA autoencoder")
    return freeze_encdec(encdec)


def load_b1_denoiser(path: Path) -> TransformerDenoiser:
    assert_b1_initialization_path(path)
    denoiser = TransformerDenoiser.from_pretrained(str(path), subfolder="unet")
    assert_complete_trainable(denoiser, EXPECTED_DENOISER_PARAMETER_COUNT)
    if denoiser.config.in_channels != LATENT_CHANNELS:
        raise AssertionError("denoiser in_channels is not %d" % LATENT_CHANNELS)
    return denoiser


def build_adamw(denoiser: torch.nn.Module, learning_rate: float) -> torch.optim.AdamW:
    trainable = [parameter for parameter in denoiser.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("denoiser has no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )


def collate_sequences(examples: Sequence[dict], tokenizer) -> dict:
    encoded = [
        torch.tensor(encode_rna_characters(example["sequence"], tokenizer), dtype=torch.long)
        for example in examples
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0),
        "index": torch.tensor([example["index"] for example in examples], dtype=torch.long),
    }


class SequenceIndexDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: Sequence[str]):
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        return {"index": index, "sequence": self.sequences[index]}


def extract_latents(encdec: EncDec, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    assert_encdec_frozen(encdec)
    with torch.no_grad():
        latent = encdec.get_latent(input_ids=input_ids, attention_mask=attention_mask)
    if tuple(latent.shape[1:]) != (LATENT_QUERIES, LATENT_CHANNELS):
        raise AssertionError("latent shape %s is not [B, 32, 160]" % (tuple(latent.shape),))
    return latent.float()


def epsilon_mse_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """Float32 denoiser sample for the epsilon MSE.

    Accelerate 0.25.0 wraps a prepared forward in autocast and then converts
    FP16/BF16 outputs back to FP32. A prediction is therefore not required to
    arrive as FP16 or BF16. BF16 is cast with float(), never via float16.
    """
    if not prediction.is_floating_point():
        raise TypeError("denoiser prediction must be floating point, got %s" % (prediction.dtype,))
    return prediction.float()


def prepare_noisy_latent(scheduler: DDIMScheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    """DDIM add_noise in float32. Mixed precision is not applied here."""
    latents = latents.float()
    noise = noise.float()
    noisy = scheduler.add_noise(latents, noise, timesteps)
    if noisy.dtype != torch.float32 or noise.dtype != torch.float32 or latents.dtype != torch.float32:
        raise AssertionError("add_noise arithmetic must stay float32")
    return noisy, noise


def denoising_mse(model, scheduler, latents, noise, timesteps, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Epsilon MSE in float32 against the sampled noise.

    The prepared denoiser is called as-is so Accelerate autocast, when enabled,
    applies inside that forward. Scheduler arithmetic and the MSE stay float32.
    """
    if dtype != torch.float32:
        raise ValueError(
            "scheduler, epsilon target and MSE stay float32; got %s. "
            "Mixed precision is Accelerate autocast on the prepared denoiser."
            % (dtype,)
        )
    if tuple(latents.shape[1:]) != (LATENT_QUERIES, LATENT_CHANNELS):
        raise AssertionError("expected latent [B, 32, 160], got %s" % (tuple(latents.shape),))
    noise = noise.to(device=latents.device)
    timesteps = timesteps.to(device=latents.device)
    noisy, target = prepare_noisy_latent(scheduler, latents, noise, timesteps)
    prediction = epsilon_mse_prediction(model(noisy, timesteps).sample)
    if prediction.shape != target.shape:
        raise AssertionError("prediction shape %s != target shape %s" % (tuple(prediction.shape), tuple(target.shape)))
    if prediction.dtype != torch.float32 or target.dtype != torch.float32 or noisy.dtype != torch.float32:
        raise AssertionError("noise, noisy latent, prediction and target must be float32 for the MSE")
    return F.mse_loss(prediction, target)


def sample_train_diffusion(batch_size: int, generator: torch.Generator, num_train_timesteps: int, device: torch.device):
    """Fresh training timestep and float32 noise. Advances only this generator."""
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,), generator=generator, dtype=torch.long)
    noise = torch.randn(batch_size, LATENT_QUERIES, LATENT_CHANNELS, generator=generator)
    if noise.dtype != torch.float32:
        raise AssertionError("training epsilon must be generated in float32")
    return timesteps.to(device), noise.to(device)


def build_fixed_validation_draws(
    sequences: Sequence[str],
    validation_seed: int,
    draws_per_sequence: int,
    num_train_timesteps: int,
) -> ValidationDraws:
    """Materialise the validation (t, epsilon) problem once.

    A CPU generator seeded with validation_seed walks sequences in file order,
    then draws. The tensors are stored and reused. Rebuilding with the same
    seed and sequences reproduces them. The global RNG is not used.
    """
    if draws_per_sequence < 1:
        raise ValueError("validation draws per sequence must be positive")
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(validation_seed))
    n_sequences = len(sequences)
    timesteps = torch.empty(n_sequences, draws_per_sequence, dtype=torch.long)
    noise = torch.empty(n_sequences, draws_per_sequence, LATENT_QUERIES, LATENT_CHANNELS)
    for sequence_index in range(n_sequences):
        for draw_index in range(draws_per_sequence):
            timesteps[sequence_index, draw_index] = torch.randint(
                0, num_train_timesteps, (1,), generator=generator, dtype=torch.long
            )
            noise[sequence_index, draw_index] = torch.randn(
                LATENT_QUERIES, LATENT_CHANNELS, generator=generator
            )
    payload = "\n".join(sequences)
    return ValidationDraws(
        seed=int(validation_seed),
        draws_per_sequence=draws_per_sequence,
        num_train_timesteps=num_train_timesteps,
        sequences_sha256=sha256_text(payload),
        timesteps=timesteps,
        noise=noise,
    )


def validation_denoising_loss(
    model,
    encdec: EncDec,
    scheduler: DDIMScheduler,
    loader: torch.utils.data.DataLoader,
    draws: ValidationDraws,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Held-out denoising MSE on the same model object training just used.

    Pass the prepared denoiser. Do not unwrap it only to evaluate it.
    No backward and no optimiser step.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    counted = 0
    try:
        with torch.no_grad():
            for batch in loader:
                latent = extract_latents(encdec, batch["input_ids"], batch["attention_mask"])
                indexes = batch["index"].tolist()
                for draw_index in range(draws.draws_per_sequence):
                    timesteps = draws.timesteps[indexes, draw_index]
                    noise = draws.noise[indexes, draw_index]
                    loss = denoising_mse(model, scheduler, latent, noise, timesteps, dtype)
                    batch_size = latent.shape[0]
                    total += float(loss.detach()) * batch_size
                    counted += batch_size
    finally:
        model.train(was_training)
    if counted == 0:
        raise RuntimeError("VALIDATION loader produced no examples")
    return total / counted


def select_checkpoint_epoch(validation_losses: Sequence[float]) -> int:
    """Lowest fixed-draw VALIDATION loss. Ties keep the earliest epoch.

    Generated-sample guardrails are a later gate. This trainer does not generate.
    """
    if not validation_losses:
        raise ValueError("checkpoint selection requires at least one validation loss")
    return min(range(len(validation_losses)), key=lambda epoch: (validation_losses[epoch], epoch))


def format_learning_rate(learning_rate: float) -> str:
    return format(learning_rate, ".0e").replace("e-0", "e-").replace("e+0", "e+")


def epoch_checkpoint_name(train_sha256: str, learning_rate: float, seed: int, epoch: int, git_sha: str) -> str:
    return (
        "D1_generic-diffusion__data-%s__lr-%s__ebs-%d__seed-%d__epoch-%02d__git-%s"
        % (train_sha256[:8], format_learning_rate(learning_rate), EFFECTIVE_BATCH_SIZE, seed, epoch, git_sha[:7])
    )


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=str(REPO_ROOT), text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    return bool(status.strip())


def build_run_metadata(args, plan: UpdatePlan, denoiser_report: dict, encdec_frozen: bool, train_sha: str, validation_sha: str, draws: ValidationDraws) -> dict:
    return {
        "condition": "D1_mres_curated_aptamer_diffusion_adapted",
        "rnagenesis_git": git_head(),
        "working_tree_dirty": git_dirty(),
        "upstream_reference": "6bc641acfb16aa2dfcbfd33b50fe5085150777a9",
        "b1_checkpoint": str(B1_DIFFUSION_DIR),
        "encdec_checkpoint": str(args.encdec_checkpoint),
        "train_path": str(args.train_data),
        "validation_path": str(args.validation_data),
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "train_examples": plan.train_examples,
        "validation_cores": int(draws.timesteps.shape[0]),
        "microbatch": plan.microbatch,
        "gradient_accumulation": plan.gradient_accumulation,
        "effective_batch": plan.effective_batch,
        "dataloader_batches_per_epoch": plan.dataloader_batches_per_epoch,
        "optimizer_updates_per_epoch": plan.optimizer_updates_per_epoch,
        "epochs": plan.epochs,
        "total_optimizer_updates": plan.total_optimizer_updates,
        "warmup_updates": plan.warmup_updates,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "adam_beta1": ADAM_BETA1,
        "adam_beta2": ADAM_BETA2,
        "adam_epsilon": ADAM_EPSILON,
        "adam_weight_decay": ADAM_WEIGHT_DECAY,
        "lr_scheduler": "cosine",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "gradient_clip_provenance": "implementation plan section 11.1; upstream AdamW clip adopted explicitly",
        "mixed_precision": args.mixed_precision,
        "denoiser_precision": "accelerate_prepared_model_autocast",
        "accelerate_autocast_dtype": (
            "disabled"
            if args.mixed_precision == "no"
            else str(diffusion_compute_dtype(args.mixed_precision)).replace("torch.", "")
        ),
        "scheduler_noise_target_loss_dtype": "float32",
        "bf16_mapped_to_fp16": False,
        "accumulation_weighting": "mean_loss_times_microbatch_over_window",
        "seed": args.seed,
        "train_loader_seed": args.seed,
        "train_diffusion_seed": args.seed,
        "validation_seed": args.validation_seed,
        "validation_draws_per_sequence": draws.draws_per_sequence,
        "validation_sequence_sha256": draws.sequences_sha256,
        "validation_timestep_sha256": sha256_text(draws.timesteps.cpu().numpy().tobytes().hex()),
        "validation_noise_sha256": sha256_text(draws.noise.cpu().numpy().tobytes().hex()),
        "denoiser_parameter_count": denoiser_report["total"],
        "denoiser_trainable_parameter_count": denoiser_report["trainable"],
        "encdec_frozen": encdec_frozen,
        "generation_weights": "raw",
        "ema": False,
        "ema_note": "B1 release layout has no unet_ema; D1 does not create a new EMA state",
        "checkpoint_rule": "lowest fixed-draw VALIDATION denoising loss; earliest epoch on ties; generation guardrails not applied here",
        "external_contract_verifier": EXTERNAL_CONTRACT_VERIFIER,
        "preflight": bool(args.preflight),
        "fixture": bool(args.fixture),
    }


def save_diffusers_checkpoint(denoiser, scheduler, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = DDIMPipeline1D(unet=denoiser, scheduler=scheduler)
    pipeline.save_pretrained(str(directory))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRes D1 latent-diffusion trainer")
    parser.add_argument("--train_data", type=Path, required=True, help="Frozen TRAIN text: corpus_final.txt")
    parser.add_argument("--validation_data", type=Path, required=True, help="Frozen VALIDATION CSV: ft_validation.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encdec_checkpoint", type=Path, default=ENCDEC_DIR)
    parser.add_argument("--pretrained_ckpts", type=Path, default=B1_DIFFUSION_DIR)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED, help="TRAIN loader, training diffusion stream, and Accelerator process seed")
    parser.add_argument("--validation_seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--validation_draws", type=int, default=DEFAULT_VALIDATION_DRAWS)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--preflight", action="store_true", help="Check the run and exit before any optimiser step")
    parser.add_argument("--fixture", action="store_true", help="Synthetic inputs. Disables frozen SHA256 equality. Not for biological data.")
    args = parser.parse_args(argv)
    if args.seed == args.validation_seed:
        parser.error("--validation_seed must differ from --seed so the fixed validation draws are not the training stream")
    return args


def _loader(sequences: Sequence[str], batch_size: int, shuffle: bool, seed: Optional[int], tokenizer, num_workers: int):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        SequenceIndexDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
        collate_fn=lambda examples: collate_sequences(examples, tokenizer),
    )


def run_training(args: argparse.Namespace) -> dict:
    assert_diffusers_normalization_patch()
    assert_output_isolated(args.output, (args.train_data, args.validation_data))
    assert_distinct_roles(args.train_data, args.validation_data)
    enforce_hash = not args.fixture
    train_sequences = read_train_sequences(args.train_data, enforce_hash)
    validation_sequences = read_validation_cores(args.validation_data, enforce_hash)
    if enforce_hash:
        assert_frozen_role_counts(len(train_sequences), len(validation_sequences))
    plan = build_update_plan(
        len(train_sequences),
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.num_epochs,
    )
    assert_b1_initialization_path(args.pretrained_ckpts)
    if args.encdec_checkpoint.resolve() != ENCDEC_DIR.resolve():
        raise ValueError("D1 EncDec checkpoint must be configs/rnagenesis/autoencoder")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("D1 is single-GPU only")
    set_seed(args.seed)
    if diffusion_compute_dtype(args.mixed_precision) == torch.float16 and args.mixed_precision == "bf16":
        raise AssertionError("bf16 must not be mapped to fp16")

    encdec = load_frozen_encdec(args.encdec_checkpoint)
    denoiser = load_b1_denoiser(args.pretrained_ckpts)
    denoiser_report = assert_complete_trainable(denoiser, EXPECTED_DENOISER_PARAMETER_COUNT)
    scheduler = load_scheduler(args.pretrained_ckpts)
    optimizer = build_adamw(denoiser, args.learning_rate)
    assert_optimizer_membership(optimizer, denoiser, encdec)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=plan.warmup_updates,
        num_training_steps=plan.total_optimizer_updates,
    )

    tokenizer = load_tokenizer()
    train_loader = _loader(train_sequences, args.train_batch_size, True, args.seed, tokenizer, args.dataloader_num_workers)
    validation_loader = _loader(validation_sequences, args.train_batch_size, False, None, tokenizer, args.dataloader_num_workers)
    draws = build_fixed_validation_draws(
        validation_sequences,
        args.validation_seed,
        args.validation_draws,
        int(scheduler.config.num_train_timesteps),
    )
    metadata = build_run_metadata(
        args,
        plan,
        denoiser_report,
        True,
        sha256_file(assert_allowed_input_path(args.train_data)),
        sha256_file(assert_allowed_input_path(args.validation_data)),
        draws,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preflight_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print("preflight complete; no optimiser step")
        return metadata

    denoiser, optimizer, train_loader, validation_loader, lr_scheduler = accelerator.prepare(
        denoiser, optimizer, train_loader, validation_loader, lr_scheduler
    )
    encdec.to(accelerator.device)
    freeze_encdec(encdec)
    assert_optimizer_membership(optimizer, accelerator.unwrap_model(denoiser), encdec)

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed)
    validation_losses = []
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    for epoch in range(plan.epochs):
        denoiser.train()
        microbatches = list(train_loader)
        batch_sizes = [int(batch["input_ids"].shape[0]) for batch in microbatches]
        loss_weights = microbatch_loss_weights(batch_sizes, plan.gradient_accumulation)
        step_indexes = set(accumulation_step_indices(len(microbatches), plan.gradient_accumulation))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(microbatches):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            latent = extract_latents(encdec, input_ids, attention_mask)
            timesteps, noise = sample_train_diffusion(
                latent.shape[0], train_generator, int(scheduler.config.num_train_timesteps), latent.device
            )
            loss = denoising_mse(denoiser, scheduler, latent, noise, timesteps)
            scale = accelerator_backward_scale(loss_weights[step], plan.gradient_accumulation)
            accelerator.backward(loss * scale)
            epoch_loss += float(loss.detach())
            if step in step_indexes:
                accelerator.clip_grad_norm_(accelerator.unwrap_model(denoiser).parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        mean_validation = validation_denoising_loss(
            denoiser, encdec, scheduler, validation_loader, draws
        )
        validation_losses.append(mean_validation)
        checkpoint_dir = args.output / epoch_checkpoint_name(
            metadata["train_sha256"], args.learning_rate, args.seed, epoch, metadata["rnagenesis_git"]
        )
        if accelerator.is_main_process:
            save_diffusers_checkpoint(accelerator.unwrap_model(denoiser), scheduler, checkpoint_dir)
        print("epoch %d train_mse %s validation_mse %s" % (epoch, epoch_loss / max(len(microbatches), 1), mean_validation))

    selected = select_checkpoint_epoch(validation_losses)
    selection = {
        "selected_epoch": selected,
        "validation_losses": validation_losses,
        "rule": metadata["checkpoint_rule"],
        "checkpoint": epoch_checkpoint_name(
            metadata["train_sha256"], args.learning_rate, args.seed, selected, metadata["rnagenesis_git"]
        ),
    }
    if accelerator.is_main_process:
        (args.output / "selected_checkpoint.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
