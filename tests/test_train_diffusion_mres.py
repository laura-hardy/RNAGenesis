"""Synthetic tests for the MRes D1 trainer. No biological files and no optimiser step."""

import csv
import inspect
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers import DDIMScheduler

import train_diffusion_mres as trainer


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class LoaderTests(unittest.TestCase):
    def test_train_and_validation_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "corpus_final.txt"
            validation = root / "ft_validation.csv"
            _write(train, "ACGU\nGGGG\n")
            with validation.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence_core", "ft_split"])
                writer.writeheader()
                writer.writerow({"sequence_core": "UUUU", "ft_split": "VALIDATION"})
                writer.writerow({"sequence_core": "UUUU", "ft_split": "VALIDATION"})
                writer.writerow({"sequence_core": "CCCC", "ft_split": "VALIDATION"})
            trainer.assert_distinct_roles(train, validation)
            self.assertEqual(trainer.read_train_sequences(train, enforce_frozen_hash=False), ["ACGU", "GGGG"])
            self.assertEqual(trainer.read_validation_cores(validation, enforce_frozen_hash=False), ["UUUU", "CCCC"])

    def test_no_internal_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            sequences = ["A" * (index + 1) for index in range(10)]
            _write(path, "\n".join(sequences) + "\n")
            loaded = trainer.read_train_sequences(path, enforce_frozen_hash=False)
            self.assertEqual(loaded, sequences)
            self.assertEqual(len(loaded), 10)

    def test_rejects_noncanonical_and_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in (
                ("t.txt", "ACGT\n"),
                ("lower.txt", "acgu\n"),
                ("space.txt", "ACGU \n"),
                ("lead.txt", " ACGU\n"),
                ("empty.txt", "\n"),
                ("n.txt", "ACGN\n"),
            ):
                path = root / name
                _write(path, text)
                with self.assertRaises(ValueError):
                    trainer.read_train_sequences(path, enforce_frozen_hash=False)

    def test_crlf_delimiter_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_bytes(b"ACGU\r\nGGGG\r\n")
            self.assertEqual(trainer.read_train_sequences(path, enforce_frozen_hash=False), ["ACGU", "GGGG"])

    def test_duplicate_train_line_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            _write(path, "ACGU\nACGU\n")
            with self.assertRaises(ValueError):
                trainer.read_train_sequences(path, enforce_frozen_hash=False)

    def test_same_path_cannot_be_both_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus_final.txt"
            _write(path, "ACGU\n")
            with self.assertRaises(ValueError):
                trainer.assert_distinct_roles(path, path)


class RoleExclusionTests(unittest.TestCase):
    def test_ref_and_eval_names_are_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("ref.csv", "eval.csv"):
                path = root / name
                _write(path, "ACGU\n")
                with self.assertRaises(ValueError):
                    trainer.read_train_sequences(path, enforce_frozen_hash=False)

    def test_cli_has_no_ref_or_eval_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.txt"
            validation = root / "validation.csv"
            _write(train, "ACGU\n")
            _write(validation, "sequence_core\nACGU\n")
            base = ["--train_data", str(train), "--validation_data", str(validation), "--output", str(root / "out"), "--fixture"]
            for forbidden in ("--ref_data", "--eval_data", "--ref", "--eval"):
                with self.assertRaises(SystemExit):
                    trainer.parse_args(base + [forbidden, str(root / "nope.csv")])

    def test_frozen_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            _write(path, "ACGU\n")
            with self.assertRaises(ValueError):
                trainer.read_train_sequences(path, enforce_frozen_hash=True)

    def test_b0_path_is_rejected(self):
        with self.assertRaises(ValueError):
            trainer.assert_b1_initialization_path(trainer.B0_DIFFUSION_DIR)
        self.assertEqual(
            trainer.assert_b1_initialization_path(trainer.B1_DIFFUSION_DIR),
            trainer.B1_DIFFUSION_DIR.resolve(),
        )


class ArithmeticTests(unittest.TestCase):
    def test_approved_669_configurations(self):
        for microbatch, accumulation in ((8, 4), (16, 2), (32, 1)):
            plan = trainer.build_update_plan(669, microbatch, accumulation, epochs=5)
            self.assertEqual(plan.effective_batch, 32)
            self.assertEqual(plan.optimizer_updates_per_epoch, 21)
            self.assertEqual(plan.total_optimizer_updates, 105)
            self.assertEqual(plan.warmup_updates, 10)
            self.assertEqual(trainer.build_update_plan(640, microbatch, accumulation, 5).optimizer_updates_per_epoch, 20)

    def test_partial_final_accumulation_window_is_stepped(self):
        self.assertEqual(trainer.accumulation_step_indices(5, 3), [2, 4])
        self.assertEqual(trainer.optimizer_updates_per_epoch(10, 3, 2), 2)

    def test_disallowed_batch_pair(self):
        with self.assertRaises(ValueError):
            trainer.build_update_plan(669, 4, 8, 5)


class DtypeTests(unittest.TestCase):
    def test_bf16_is_not_fp16(self):
        self.assertIs(trainer.diffusion_compute_dtype("no"), torch.float32)
        self.assertIs(trainer.diffusion_compute_dtype("fp16"), torch.float16)
        self.assertIs(trainer.diffusion_compute_dtype("bf16"), torch.bfloat16)
        self.assertIsNot(trainer.diffusion_compute_dtype("bf16"), torch.float16)
        with self.assertRaises(ValueError):
            trainer.diffusion_compute_dtype("fp32")


class ObjectiveTests(unittest.TestCase):
    def test_epsilon_mse_matches_sampled_noise(self):
        scheduler = DDIMScheduler.from_pretrained(str(trainer.B1_DIFFUSION_DIR), subfolder="scheduler")

        class IdentityNoise(torch.nn.Module):
            def forward(self, hidden, timestep):
                return type("Out", (), {"sample": hidden * 0 + self.noise})()

        model = IdentityNoise()
        latents = torch.zeros(2, 32, 160)
        noise = torch.randn(2, 32, 160)
        model.noise = noise
        timesteps = torch.tensor([3, 9])
        loss = trainer.denoising_mse(model, scheduler, latents, noise, timesteps, torch.float32)
        self.assertEqual(tuple(latents.shape), (2, 32, 160))
        self.assertLess(float(loss), 1e-6)

    def test_dtype_is_applied_to_noise_and_target(self):
        scheduler = DDIMScheduler.from_pretrained(str(trainer.B1_DIFFUSION_DIR), subfolder="scheduler")

        class ReturnNoise(torch.nn.Module):
            def forward(self, hidden, timestep):
                self.seen = hidden.dtype
                return type("Out", (), {"sample": torch.zeros_like(hidden)})()

        model = ReturnNoise()
        latents = torch.zeros(1, 32, 160)
        noise = torch.zeros(1, 32, 160)
        trainer.denoising_mse(model, scheduler, latents, noise, torch.tensor([0]), torch.float32)
        self.assertIs(model.seen, torch.float32)


class FreezeAndOptimizerTests(unittest.TestCase):
    def test_frozen_assertion_and_denoiser_only_optimizer(self):
        denoiser = torch.nn.Linear(4, 4)
        encdec = torch.nn.Linear(4, 4)
        encdec.eval()
        for parameter in encdec.parameters():
            parameter.requires_grad_(False)
        trainer.assert_encdec_frozen(encdec)
        report = trainer.assert_complete_trainable(denoiser)
        self.assertEqual(report["trainable"], report["total"])
        optimizer = trainer.build_adamw(denoiser, trainer.DEFAULT_LEARNING_RATE)
        trainer.assert_optimizer_membership(optimizer, denoiser, encdec)
        self.assertEqual(optimizer.param_groups[0]["betas"], (trainer.ADAM_BETA1, trainer.ADAM_BETA2))
        self.assertEqual(optimizer.param_groups[0]["eps"], trainer.ADAM_EPSILON)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], trainer.ADAM_WEIGHT_DECAY)

    def test_frozen_or_encdec_parameter_in_optimizer_fails(self):
        denoiser = torch.nn.Linear(4, 4)
        encdec = torch.nn.Linear(4, 4)
        encdec.eval()
        for parameter in encdec.parameters():
            parameter.requires_grad_(False)
        bad = torch.optim.AdamW(list(denoiser.parameters()) + list(encdec.parameters()), lr=1e-3)
        with self.assertRaises(AssertionError):
            trainer.assert_optimizer_membership(bad, denoiser, encdec)

    def test_partially_frozen_denoiser_fails_complete_count(self):
        denoiser = torch.nn.Linear(4, 4)
        denoiser.bias.requires_grad_(False)
        with self.assertRaises(AssertionError):
            trainer.assert_complete_trainable(denoiser, expected_total=None)

    def test_b1_header_is_serialized_state_not_parameter_count(self):
        path = trainer.B1_DIFFUSION_DIR / "unet" / "diffusion_pytorch_model.safetensors"
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        serialized = 0
        buffers = 0
        buffer_names = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            product = 1
            for dim in value["shape"]:
                product *= dim
            serialized += product
            if key.endswith(".pos_embed.pe"):
                self.assertEqual(value["shape"], [1, 1024, 2048])
                buffers += product
                buffer_names.append(key)
        parameters = serialized - buffers
        self.assertEqual(
            sorted(buffer_names, key=lambda name: int(name.split(".")[1])),
            ["transformer_blocks.%d.pos_embed.pe" % index for index in range(24)],
        )
        self.assertEqual(serialized, trainer.EXPECTED_DENOISER_SERIALIZED_STATE_ELEMENT_COUNT)
        self.assertEqual(buffers, trainer.EXPECTED_DENOISER_BUFFER_ELEMENT_COUNT)
        self.assertEqual(parameters, trainer.EXPECTED_DENOISER_PARAMETER_COUNT)
        self.assertEqual(parameters + buffers, serialized)
        self.assertNotEqual(serialized, trainer.EXPECTED_DENOISER_PARAMETER_COUNT)

    def test_state_accounting_keeps_buffers_out_of_parameters_and_optimizer(self):
        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4))
                self.register_buffer("pe", torch.zeros(1, 2, 3))

        denoiser = WithBuffer()
        encdec = torch.nn.Linear(4, 4)
        encdec.eval()
        for parameter in encdec.parameters():
            parameter.requires_grad_(False)
        report = trainer.denoiser_state_report(denoiser)
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["trainable"], 4)
        self.assertEqual(report["frozen"], 0)
        self.assertEqual(report["buffer_elements"], 6)
        self.assertEqual(report["serialized_state_elements"], 10)
        self.assertEqual(report["total"] + report["buffer_elements"], report["serialized_state_elements"])
        trainer.assert_complete_trainable(denoiser, expected_total=4)
        optimizer = trainer.build_adamw(denoiser, trainer.DEFAULT_LEARNING_RATE)
        trainer.assert_optimizer_membership(optimizer, denoiser, encdec)
        optimizer_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        self.assertEqual(optimizer_ids, {id(denoiser.weight)})
        self.assertNotIn(id(denoiser.pe), optimizer_ids)

    def test_released_count_constants_partition_serialized_state(self):
        self.assertEqual(
            trainer.EXPECTED_DENOISER_PARAMETER_COUNT + trainer.EXPECTED_DENOISER_BUFFER_ELEMENT_COUNT,
            trainer.EXPECTED_DENOISER_SERIALIZED_STATE_ELEMENT_COUNT,
        )
        self.assertEqual(trainer.EXPECTED_DENOISER_PARAMETER_COUNT, 1_911_589_344)
        self.assertEqual(trainer.EXPECTED_DENOISER_BUFFER_ELEMENT_COUNT, 50_331_648)
        self.assertEqual(trainer.EXPECTED_DENOISER_SERIALIZED_STATE_ELEMENT_COUNT, 1_961_920_992)


class ValidationDrawTests(unittest.TestCase):
    def test_draws_repeat_and_training_stream_advances(self):
        sequences = ["ACGU", "GGGG", "UUUU"]
        torch.manual_seed(123)
        first = trainer.build_fixed_validation_draws(sequences, validation_seed=1, draws_per_sequence=1, num_train_timesteps=1000)
        torch.manual_seed(999)
        second = trainer.build_fixed_validation_draws(sequences, validation_seed=1, draws_per_sequence=1, num_train_timesteps=1000)
        self.assertTrue(torch.equal(first.timesteps, second.timesteps))
        self.assertTrue(torch.equal(first.noise, second.noise))
        generator = torch.Generator().manual_seed(trainer.DEFAULT_TRAIN_SEED)
        one = trainer.sample_train_diffusion(2, generator, 1000, torch.device("cpu"))
        two = trainer.sample_train_diffusion(2, generator, 1000, torch.device("cpu"))
        self.assertFalse(torch.equal(one[0], two[0]) and torch.equal(one[1], two[1]))
        third = trainer.build_fixed_validation_draws(sequences, validation_seed=1, draws_per_sequence=1, num_train_timesteps=1000)
        self.assertTrue(torch.equal(first.noise, third.noise))

    def test_validation_evaluation_does_not_backward(self):
        scheduler = DDIMScheduler.from_pretrained(str(trainer.B1_DIFFUSION_DIR), subfolder="scheduler")

        class FakeEncDec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.freeze()

            def freeze(self):
                self.eval()
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                return self

            def get_latent(self, input_ids, attention_mask):
                return torch.zeros(input_ids.shape[0], 32, 160)

        class SpyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))
                self.seen = []

            def forward(self, hidden, timestep):
                self.seen.append(timestep.detach().cpu().clone())
                return type("Out", (), {"sample": hidden * 0})()

        encdec = FakeEncDec()
        model = SpyDenoiser()
        sequences = ["ACGU", "GGGG"]
        draws = trainer.build_fixed_validation_draws(sequences, 1, 1, int(scheduler.config.num_train_timesteps))
        tokenizer = trainer.load_tokenizer()
        loader = trainer._loader(sequences, batch_size=2, shuffle=False, seed=None, tokenizer=tokenizer, num_workers=0)
        first = trainer.validation_denoising_loss(model, encdec, scheduler, loader, draws, torch.float32)
        second = trainer.validation_denoising_loss(model, encdec, scheduler, loader, draws, torch.float32)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(model.seen[0], model.seen[1]))
        self.assertIsNone(model.scale.grad)
        self.assertFalse(encdec.training)


class SelectionTests(unittest.TestCase):
    def test_lowest_loss_earliest_tie(self):
        self.assertEqual(trainer.select_checkpoint_epoch([0.4, 0.2, 0.2, 0.5]), 1)
        self.assertEqual(trainer.GRADIENT_CLIP_NORM, 1.0)


class AccumulationWeightTests(unittest.TestCase):
    def test_full_window_is_uniform(self):
        weights = trainer.microbatch_loss_weights([8, 8, 8, 8], 4)
        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25])

    def test_final_short_biological_batches(self):
        eight = trainer.microbatch_loss_weights([8, 8, 8, 5], 4)
        self.assertEqual(eight, [8 / 29, 8 / 29, 8 / 29, 5 / 29])
        sixteen = trainer.microbatch_loss_weights([16, 13], 2)
        self.assertEqual(sixteen, [16 / 29, 13 / 29])
        single = trainer.microbatch_loss_weights([29], 1)
        self.assertEqual(single, [1.0])

    def test_incomplete_final_accumulation_window_is_reweighted(self):
        sizes = [8, 8, 8, 8, 8, 5]
        windows = trainer.accumulation_windows(sizes, 4)
        self.assertEqual(windows, [[8, 8, 8, 8], [8, 5]])
        weights = trainer.microbatch_loss_weights(sizes, 4)
        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25, 8 / 13, 5 / 13])
        self.assertEqual(len(trainer.accumulation_step_indices(len(sizes), 4)), 2)

    def test_weights_reproduce_the_example_mean(self):
        sizes = [8, 8, 8, 5]
        weights = trainer.microbatch_loss_weights(sizes, 4)
        values = torch.arange(sum(sizes), dtype=torch.float64)
        offset = 0
        accumulated = 0.0
        for size, weight in zip(sizes, weights):
            accumulated += float(values[offset:offset + size].mean()) * weight
            offset += size
        self.assertAlmostEqual(accumulated, float(values.mean()), places=12)

    def test_weighted_backward_matches_full_window_gradient(self):
        torch.manual_seed(0)
        targets = torch.randn(29)
        parameter = torch.nn.Parameter(torch.tensor(0.3))
        full = ((parameter - targets) ** 2).mean()
        full.backward()
        expected = parameter.grad.detach().clone()
        parameter.grad = None
        sizes = [8, 8, 8, 5]
        weights = trainer.microbatch_loss_weights(sizes, 4)
        offset = 0
        for size, weight in zip(sizes, weights):
            chunk = targets[offset:offset + size]
            loss = ((parameter - chunk) ** 2).mean()
            (loss * weight).backward()
            offset += size
        self.assertTrue(torch.allclose(parameter.grad, expected, atol=1e-6))

    def test_accelerator_backward_net_scale_is_b_over_w(self):
        weights = trainer.microbatch_loss_weights([8, 8, 8, 5], 4)
        net = [trainer.accelerator_backward_scale(weight, 4) / 4 for weight in weights]
        self.assertEqual(net, [8 / 29, 8 / 29, 8 / 29, 5 / 29])

    def test_669_still_has_21_updates(self):
        expected_tails = {8: [8, 8, 8, 5], 16: [16, 13], 32: [29]}
        for microbatch, accumulation in ((8, 4), (16, 2), (32, 1)):
            full, remainder = divmod(669, microbatch)
            sizes = [microbatch] * full + ([remainder] if remainder else [])
            windows = trainer.accumulation_windows(sizes, accumulation)
            self.assertEqual(len(windows), 21)
            self.assertEqual(windows[-1], expected_tails[microbatch])
            self.assertEqual(trainer.optimizer_updates_per_epoch(669, microbatch, accumulation), 21)


class OutputIsolationTests(unittest.TestCase):
    def test_output_under_input_parent_is_rejected_and_independent_output_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "development"
            protected.mkdir()
            train = protected / "corpus_final.txt"
            validation = protected / "ft_validation.csv"
            train.write_text("ACGU\n", encoding="utf-8")
            validation.write_text("sequence_core\nACGU\n", encoding="utf-8")
            nested = protected / "runs" / "d1"
            with self.assertRaises(ValueError):
                trainer.assert_output_isolated(protected, (train, validation))
            with self.assertRaises(ValueError):
                trainer.assert_output_isolated(nested, (train, validation))
            self.assertFalse(nested.exists())
            independent = root / "rnagenesis-mres-runs" / "d1"
            resolved = trainer.assert_output_isolated(independent, (train, validation))
            self.assertEqual(resolved, independent.resolve())
            self.assertFalse(independent.exists())


class DiffusersPatchTests(unittest.TestCase):
    def test_imported_normalization_matches_required_patch(self):
        digest = trainer.assert_diffusers_normalization_patch()
        self.assertEqual(digest, trainer.DIFFUSERS_NORMALIZATION_SHA256)
        source = trainer.diffusers_normalization_source_path()
        self.assertEqual(source.name, "normalization.py")
        lookup = inspect.getsource(trainer.diffusers_normalization_source_path)
        self.assertIn("getsourcefile", lookup)
        self.assertNotIn("site-packages", lookup)

    def test_mismatched_normalization_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normalization.py"
            path.write_text("not the required patch\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                trainer.assert_diffusers_normalization_patch(path)


class MixedPrecisionPolicyTests(unittest.TestCase):
    def test_bf16_prediction_is_scored_in_fp32_without_casting_inputs(self):
        scheduler = DDIMScheduler.from_pretrained(str(trainer.B1_DIFFUSION_DIR), subfolder="scheduler")

        class ReturnBF16(torch.nn.Module):
            def forward(self, hidden, timestep):
                self.input_dtype = hidden.dtype
                return type("Out", (), {"sample": torch.zeros_like(hidden, dtype=torch.bfloat16)})()

        model = ReturnBF16()
        latents = torch.zeros(1, 32, 160)
        noise = torch.zeros(1, 32, 160)
        loss = trainer.denoising_mse(model, scheduler, latents, noise, torch.tensor([0]), torch.float32)
        self.assertIs(model.input_dtype, torch.float32)
        self.assertIs(trainer.epsilon_mse_prediction(torch.zeros(1, dtype=torch.bfloat16)).dtype, torch.float32)
        self.assertIs(trainer.epsilon_mse_prediction(torch.zeros(1, dtype=torch.float16)).dtype, torch.float32)
        self.assertLess(float(loss), 1e-6)
        with self.assertRaises(ValueError):
            trainer.denoising_mse(model, scheduler, latents, noise, torch.tensor([0]), torch.float16)
        with self.assertRaises(ValueError):
            trainer.denoising_mse(model, scheduler, latents, noise, torch.tensor([0]), torch.bfloat16)

    def test_validation_call_uses_the_prepared_denoiser(self):
        source = inspect.getsource(trainer.run_training)
        self.assertIn("validation_denoising_loss(\n            denoiser,", source)
        self.assertNotIn("validation_denoising_loss(\n            accelerator.unwrap_model", source)
        self.assertIn("accelerator.unwrap_model(denoiser)", source)


class FrozenRoleCountTests(unittest.TestCase):
    def test_expected_frozen_counts(self):
        trainer.assert_frozen_role_counts(669, 95)
        with self.assertRaises(ValueError):
            trainer.assert_frozen_role_counts(668, 95)
        with self.assertRaises(ValueError):
            trainer.assert_frozen_role_counts(669, 94)

    def test_fixture_loader_does_not_require_frozen_counts(self):
        source = inspect.getsource(trainer.run_training)
        self.assertIn("if enforce_hash:\n        assert_frozen_role_counts", source)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            _write(path, "ACGU\n")
            self.assertEqual(trainer.read_train_sequences(path, enforce_frozen_hash=False), ["ACGU"])


if __name__ == "__main__":
    unittest.main()
