# ViRL39K preprocessing

The converter in `virl39k.py` prepares the pinned ViRL39K release for VERL
multimodal RL training. It keeps the official boxed answers, repairs only
unambiguous missing image placeholders, and creates a deterministic validation
split without normalized-question leakage.

## Source snapshot

- Dataset: `TIGER-Lab/ViRL39K`
- Revision: `812ec617dea4bc8a4e751663b88e4ebb7de4d00e`
- `39Krelease.parquet` SHA-256:
  `ea1655368fd3774d37baf54c65aef345b0766f4c80d6aeac91a311fa7e4fba92`
- `images.zip` SHA-256:
  `49e6485a02815430fe4a468a25c03c3514adf8c2b68135068fc36c7656d8ad27`

Expected raw layout:

```text
/data/datasets/virl39k/raw/
├── 39Krelease.parquet
└── images/
```

## Convert

Run inside the VERL container:

```bash
python /verl/examples/data_preprocess/virl39k.py \
  --parquet /data/datasets/virl39k/raw/39Krelease.parquet \
  --image-dir /data/datasets/virl39k/raw/images \
  --output-dir /data/datasets/virl39k/processed \
  --val-size 1800
```

The output contains `train.parquet`, `val.parquet`, their
`*_single_image.parquet` subsets, and `audit.json`. A split may exceed the
requested validation size slightly because all duplicate normalized questions
remain in the same split.

Use the single-image files while the training configuration contains
`limit_mm_per_prompt.image=1`. The full files retain the multi-image examples
for future rollout support.

The data source is `TIGER-Lab/ViRL39K`; VERL dispatches it to the ViRL39K
rule reward in `verl/utils/reward_score/virl39k.py`. The reward uses
`mathruler` for NPU-environment compatibility and scores boxed-answer accuracy
plus boxed-output format.

## Local OpenPangu compatibility

The local `openPangu-VL-7B/processor_openpangu_vl.py` must import NumPy as
`np`. The processor uses `np` when producing multimodal token type IDs, but the
downloaded model snapshot omitted that import.
