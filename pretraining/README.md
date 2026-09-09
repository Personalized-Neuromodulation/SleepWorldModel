# Pretraining

This package owns training methods around a reusable backbone. A method may own multiple backbone instances, projectors or predictors, target construction, parameter-update rules, and losses.

```text
pretraining/
├── jepa/              # online/target backbone, predictor, stop-gradient and EMA
├── contrastive/       # paired views, shared backbone, projector and pair rules
├── masked_code/       # masked backbone features predict external codebook ids
└── objectives/        # reusable objective terms such as SIGReg
```

JEPA calls an online backbone on the context view and a target backbone on the clean target view. Contrastive learning calls one shared backbone for each view. Masked-code pretraining calls the backbone for context features and `tokenization.codebook` for target ids. None of these methods is implemented inside `backbone`.

The existing `world_model.ssl` package remains the current minimal baseline until the new backbone interface is implemented and its migration is reviewed.
