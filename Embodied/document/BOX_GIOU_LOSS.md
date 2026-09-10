# Auxiliary box GIoU during stream-packed MTP training

`locany_finetune_magi_stream.py` uses `total_loss = ce_loss + lambda_box * giou_loss`.
The default `--lambda_box` is **0.1**; `--lambda_box 0` disables GIoU. The LoRA
visual-prompt shell wrapper exposes the same setting as `LAMBDA_BOX`. The training
argument is saved in the model config together with the explicit coordinate token
IDs. Pass the desired weight again when launching a resumed training run.

Fused CE is unchanged: **all supervised tokens, including coordinates, MTP
replicas, and supervised `<null>` tokens remain in CE**. Coordinate weights are
not applied to CE. The existing unused `loss_weight` behavior is unchanged.

## Alignment and metadata

Before MTP expansion, the dataset finds exact supervised
`<box><x1><y1><x2><y2></box>` sequences in canonical assistant labels. It emits
`bbox_coord_positions` (`[N, 4]`, raw label indices) and `gt_boxes` (`[N, 4]`,
numeric xyxy coordinates). The repository defines `<0>` through `<1000>` in
`number_tokens_list`; its visual-prompt cropping code uses the same 0–1000 scale.
The tokenizer is checked for unique single coordinate tokens at startup. Numeric
values come from list order, never from vocabulary IDs or an assumed contiguous
ID range.

Only these original boxes receive GIoU. The MTP-generated targets are still
trained by CE. Packing offsets box indices by the previous concatenated input
length, including its MTP suffix. The collator checks bounds and sample
boundaries. Reset/repeated `position_ids` are not used for box indexing.

The model selects `hidden_states[0, bbox_coord_positions - 1]`, matching CE's
`hidden_states[:, :-1]` versus `labels[:, 1:]` shift. It projects only the 1001
coordinate rows of the existing LM head (and bias if present), computes FP32
softmax and expected coordinates, sorts predicted corners, and normalizes both
predictions and targets by 1000. GIoU uses the public
`torchvision.ops.generalized_box_iou_loss` API with `reduction="sum"` per chunk,
followed by division by the total box count. There is no custom GIoU formula or
all-pairs box matrix. Projection is processed in
chunks of 128 boxes; no full sequence-by-vocabulary logits are added.

Ignored, incomplete, non-box, or non-positive-area GT structures are skipped.
No valid boxes produces zero auxiliary loss. Image token/feature mismatches
retain the existing behavior of zeroing the entire forward's loss. Attention,
MTP construction, packing/resume state, and inference code are unchanged.

## Logging and distributed behavior

Training logs expose `ce_loss`, `giou_loss`, `total_loss`, and `num_valid_boxes`.
These are means over forward calls and ranks during each logging interval;
`num_valid_boxes` is the mean canonical box count per rank/forward, not a total.
GIoU is averaged over valid boxes within each forward. Normal distributed
gradient reduction and gradient accumulation remain in the Trainer/DeepSpeed;
there is no global box-count reweighting. Ranks without boxes contribute zero
GIoU. Metrics are detached only after constructing the differentiable total.

The default LoRA wrapper trains LLM adapters and the visual MLP, with frozen
base language/vision weights. A frozen LM head still propagates GIoU gradients
to hidden states and their trainable upstream adapters. Other training callers
must supply canonical metadata when `lambda_box > 0`; missing metadata fails
explicitly rather than silently disabling supervision.

## CPU validation

From `Embodied/`, with matching PyTorch/torchvision builds and NumPy installed:

```bash
python -m unittest discover -s tests -p test_box_loss.py -v
```

Tests execute the production dataset/MTP, merge, collator, and model-forward
method bodies in isolation, without importing CUDA-only Magi/Liger dependencies.
The forward integration test substitutes a small CPU reference CE for the fused
CUDA kernel. GIoU-only backward checks cover trainable LM-head rows and upstream
low-rank adapter matrices with a frozen head. They do not instantiate a complete
LocateAnything checkpoint or the PEFT wrapper. Full CUDA/MagiAttention/Liger,
multi-rank DeepSpeed, and checkpoint-resume execution require the training
environment and are not established by these CPU tests.

## Library dependencies

`torchvision` is an explicit project dependency. Use a torchvision build matching
the training environment's PyTorch version and CPU/CUDA variant; the CPU test
environment is independent of the GPU training environment. The implementation
calls [torchvision's GIoU loss](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html)
directly, with its default epsilon of `1e-7`. Its epsilon handling may produce
small numerical differences from the previous handwritten formula, including a
tiny nonzero loss for identical boxes.

Restricted projection and soft coordinate expectation use PyTorch's `F.linear`,
`softmax`, and tensor reductions. Canonical-token selection and packing offsets
remain repository-specific glue: a generic bbox library cannot infer these MTP
label boundaries. Fused CE continues to use the existing Liger implementation.
