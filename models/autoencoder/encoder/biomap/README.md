# Vendored biomap / xTrimoPGLM dependency

Source repository: `Zaixi/RNAGenesis`

Pinned revision: `d8a42130984cbf04f6a5e16a3aa0c0d6578036a8`

Forensic source directory used for this vendor:
`/home/laura/rnagenesis-external/Zaixi-RNAGenesis-d8a4213`

The reconstruction call convention is the human-approved record in
`aptamer_mres_context/docs/fine_tuning/biomap_reconstruction_decision.md`.
File hashes are recorded in
`aptamer_mres_context/docs/fine_tuning/model_asset_manifest.md`.

## Files copied from the pinned revision

- `config.json`
- `configuration_xtrimopglm.py` (unchanged)
- `modeling_xtrimopglm.py` (import-compatibility edits only; see below)
- `tokenization_xtrimopglm.py` (unchanged)
- `tokenizer.model` (unchanged)
- `tokenizer_config.json` (unchanged)
- `special_tokens_map.json` (unchanged)

External Hugging Face weights were not vendored. `pytorch_model.bin` from
`Zaixi/RNAGenesis` is not present here and must not be downloaded for this
path. Encoder weights are the released RNAGenesis `EncDec` tensors
`encoder.*` in `configs/rnagenesis/autoencoder/pytorch_model.bin`.

## Deviations from the pinned `modeling_xtrimopglm.py`

`xTrimoPGLMModel.forward` is unchanged.

1. `import torch, deepspeed` is now a guarded import. `get_checkpoint_fn()`
   uses DeepSpeed checkpointing only when that package is installed and
   configured. Otherwise it uses `torch.utils.checkpoint`, which is the
   pinned function's non-DeepSpeed branch. DeepSpeed is not installed.
   The frozen D1 eval path does not enable gradient checkpointing, so this
   function is not on the approved latent path.

2. `from .quantization import quantize` is guarded because `quantization.py`
   is not in the pinned revision. If quantization is actually requested, the
   fallback raises. It does not invent a quantizer. The pinned config has
   `quantization_bit = 0`. D1 builds `xTrimoPGLMModel`, not
   `xTrimoPGLMForMaskedLM`, which is the only class that calls `quantize`
   from `__init__`.

No other pinned source lines were edited.
