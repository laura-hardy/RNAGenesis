"""Synthetic checks for the restored biomap EncDec latent path.

Uses uppercase A/C/G/U strings only. Does not read biological datasets,
build an optimiser, or select a GPU.
"""

import os
import unittest

import torch

from models.autoencoder.encdec import DataCollatorEncDec, EncDec
from models.autoencoder.encoder.biomap import (
    PAD_TOKEN_ID,
    encode_rna_characters,
    load_tokenizer,
    xTrimoPGLMConfig,
)
from models.autoencoder.encoder.biomap.rna_tokens import _BIOMAP_DIR

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENCDEC_DIR = os.path.join(REPO_ROOT, "configs", "rnagenesis", "autoencoder")
WEIGHTS = os.path.join(ENCDEC_DIR, "pytorch_model.bin")


def _encode_batch(sequences):
    tokenizer = load_tokenizer()
    ids = [torch.tensor(encode_rna_characters(seq, tokenizer), dtype=torch.long) for seq in sequences]
    collator = DataCollatorEncDec(
        num_query_tokens=32,
        data_type="rna",
        rna_encoder_type="biomap",
        rna_tokenizer_config="tokenizer_rna_v2",
    )
    # The collator also expects decoder ids. Supply a minimal decoder stub
    # token list so padding behaviour for the encoder side can be checked
    # without invoking generation.
    instances = []
    for row in ids:
        instances.append({
            "input_ids": row.tolist(),
            "decoder_input_ids": [0],
        })
    batch = collator(instances)
    return batch["input_ids"], batch["attention_mask"]


class BiomapDependencyTests(unittest.TestCase):
    def test_pinned_encoder_config(self):
        config = xTrimoPGLMConfig.from_json_file(os.path.join(_BIOMAP_DIR, "config.json"))
        self.assertEqual(config.padded_vocab_size, 128)
        self.assertEqual(config.vocab_size, 128)
        self.assertEqual(config.hidden_size, 1280)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.num_attention_heads, 20)
        self.assertEqual(config.quantization_bit, 0)
        self.assertFalse(config.is_causal)

    def test_character_ids_fail_closed(self):
        tokenizer = load_tokenizer()
        self.assertEqual(encode_rna_characters("ACGU", tokenizer), [4, 3, 2, 1])
        longer = "ACGUACGUACGU"
        self.assertEqual(
            encode_rna_characters(longer, tokenizer),
            tokenizer.convert_tokens_to_ids(longer),
        )
        self.assertEqual(encode_rna_characters(longer, tokenizer), [4, 3, 2, 1] * 3)
        self.assertEqual(tokenizer.token_to_id("<pad>"), PAD_TOKEN_ID)
        self.assertEqual(PAD_TOKEN_ID, 0)
        # The string form of the pad token is not the padding id source.
        self.assertNotEqual(tokenizer.convert_tokens_to_ids("<pad>"), [0])
        # No BOS/CLS/EOS is inserted by the character-wise mapper.
        ids = encode_rna_characters("ACGU", tokenizer)
        self.assertEqual(len(ids), len("ACGU"))
        self.assertEqual(len(encode_rna_characters(longer, tokenizer)), len(longer))
        for special in ("<bos>", "<cls>", "<eos>"):
            special_id = tokenizer.token_to_id(special)
            self.assertNotIn(special_id, ids)
            self.assertNotIn(special_id, encode_rna_characters(longer, tokenizer))
        invalid = ("", "T", "ACGT", "acgu", "ACGN", " ACGU", "ACGU ", "A C G U")
        for sequence in invalid:
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    encode_rna_characters(sequence, tokenizer)
        with self.assertRaises(ValueError):
            encode_rna_characters(None, tokenizer)

    def test_collator_pad_and_mask(self):
        input_ids, attention_mask = _encode_batch(["ACGU", "ACGUACGU"])
        self.assertEqual(tuple(input_ids.shape), (2, 8))
        self.assertTrue(torch.equal(input_ids[0, :4], torch.tensor([4, 3, 2, 1])))
        self.assertTrue(torch.equal(input_ids[0, 4:], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(attention_mask, input_ids.ne(0)))


class BiomapWeightTests(unittest.TestCase):
    def test_encoder_keys_match_released_checkpoint(self):
        """Compare key sets and shapes on the meta device.

        The checkpoint zip is read onto meta tensors so the 3.4 GB storages
        are not materialized beside a second encoder copy.
        """
        config = xTrimoPGLMConfig.from_json_file(os.path.join(_BIOMAP_DIR, "config.json"))
        with torch.device("meta"):
            encoder = __import__(
                "models.autoencoder.encoder.biomap",
                fromlist=["xTrimoPGLMModel"],
            ).xTrimoPGLMModel(config, empty_init=False)
            model_sd = encoder.state_dict()
        ckpt = torch.load(WEIGHTS, map_location="meta")
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        released = {
            key[len("encoder."):]: value
            for key, value in ckpt.items()
            if key.startswith("encoder.")
        }
        model_keys = set(model_sd.keys())
        released_keys = set(released.keys())
        self.assertEqual(model_keys, released_keys)
        self.assertGreater(len(model_keys), 0)
        for key in sorted(model_keys):
            self.assertEqual(tuple(model_sd[key].shape), tuple(released[key].shape), key)
            self.assertEqual(model_sd[key].dtype, released[key].dtype, key)
        del ckpt
        del released
        del model_sd


class BiomapLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EncDec.from_pretrained(ENCDEC_DIR)
        cls.model.freeze()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_freeze(self):
        self.assertFalse(self.model.training)
        for name, param in self.model.named_parameters():
            self.assertFalse(param.requires_grad, name)

    def test_latent_shapes_do_not_apply_transform(self):
        sequences = ["ACGUACGUACGUACGUACGU", "ACGU"]
        input_ids, attention_mask = _encode_batch(sequences)
        seen = {}

        enc_handle = self.model.encoder.register_forward_hook(
            lambda module, inputs, output: seen.__setitem__(
                "encoder_in", tuple(inputs[0].shape)
            ) or seen.__setitem__("encoder_out", tuple(output.last_hidden_state.shape))
        )

        def _qt_hook(module, args, kwargs, output):
            seen["qt_hidden"] = tuple(kwargs["encoder_hidden_states"].shape)
            seen["qt_mask"] = tuple(kwargs["encoder_attention_mask"].shape)
            seen["qt_out"] = tuple(output.last_hidden_state.shape)

        qt_handle = self.model.qt.register_forward_hook(_qt_hook, with_kwargs=True)

        def _transform_forbidden(*args, **kwargs):
            raise AssertionError("get_latent applied transform")

        original_transform = self.model.transform.forward
        self.model.transform.forward = _transform_forbidden
        try:
            with torch.no_grad():
                latent = self.model.get_latent(input_ids, attention_mask)
        finally:
            enc_handle.remove()
            qt_handle.remove()
            self.model.transform.forward = original_transform

        self.assertEqual(seen["encoder_in"], (2, 20))
        self.assertEqual(seen["encoder_out"], (20, 2, 1280))
        self.assertEqual(seen["qt_hidden"], (2, 20, 1280))
        self.assertEqual(seen["qt_mask"], (2, 20))
        self.assertEqual(seen["qt_out"], (2, 32, 160))
        self.assertEqual(tuple(latent.shape), (2, 32, 160))
        self.assertEqual(latent.shape[-1], 160)
        self.assertNotEqual(latent.shape[-1], self.model.decoder.config.hidden_size)

    def test_determinism(self):
        input_ids, attention_mask = _encode_batch(["ACGUACGUACGUACGUACGU"])
        with torch.no_grad():
            first = self.model.get_latent(input_ids, attention_mask)
            second = self.model.get_latent(input_ids, attention_mask)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
