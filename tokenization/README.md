# Tokenization

`tokenization` converts clean waveforms or continuous teacher features into discrete representations. It is separate from the dataloader and the continuous backbone because a learnable codebook has parameters, update rules, and checkpoint state.

```text
tokenization/
└── codebook/          # quantizer/tokenizer and typed CodebookOutput
```

The default masked-code path uses the codebook only on the clean target branch:

```text
clean target ──> codebook ──> code_ids
masked input ──> backbone ──> prediction head ──> code_logits
```

The output contract must include code ids, validity, layout, codebook version, and any trainable-quantizer auxiliary values. If code ids are generated offline, the dataloader may read them but does not train or update the codebook.
