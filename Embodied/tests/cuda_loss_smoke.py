"""CUDA integration smoke test: real PEFT + repository fused CE + box GIoU.

No checkpoint downloads, MagiAttention, or full LocateAnything model required.
Run from Embodied: python tests/cuda_loss_smoke.py --steps 50
"""
import argparse
import importlib.metadata
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    import torch
    from torch import nn
    from torch.nn import functional as F
    from peft import LoraConfig, get_peft_model

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eaglevl.train.box_loss import canonical_box_metadata, coordinate_giou_loss
    from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss

    if not torch.cuda.is_available():
        raise RuntimeError('This test requires a CUDA GPU; select GPU runtime in Colab.')
    for package in ('torch', 'torchvision', 'peft', 'transformers', 'liger-kernel', 'triton'):
        print(f'{package}: {importlib.metadata.version(package)}', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    class TinyDecoder(nn.Module):
        # A small hidden-state producer, not a pretrained language model.
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 64)
            self.v_proj = nn.Linear(64, 64)

        def forward(self, x):
            return self.v_proj(torch.tanh(self.q_proj(x)))

    model = get_peft_model(TinyDecoder().cuda(), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias='none',
        target_modules=['q_proj', 'v_proj'],
    ))
    model.train()
    head = nn.Linear(64, 2048, bias=False).cuda().requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, _ in trainable)
    model.print_trainable_parameters()
    initial = {n: p.detach().clone() for n, p in trainable}
    head_initial = head.weight.detach().clone()

    x = torch.randn(1, 64, 32, device='cuda')
    ids = list(range(100, 1101))
    labels = torch.full((1, 64), 12, dtype=torch.long, device='cuda')
    labels[:, 0] = -100
    labels[:, 30:34] = -100
    labels[0, 5:11] = torch.tensor([7, 200, 300, 800, 900, 8], device='cuda')
    labels[0, 20:26] = torch.tensor([7, 150, 180, 1050, 1080, 8], device='cuda')
    meta = canonical_box_metadata(labels[0], ids, 7, 8)
    assert meta['gt_boxes'].shape == (2, 4)
    targets = labels[:, 1:].contiguous().reshape(-1)
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')

    # Tiny full logits are materialized ONLY here, as a numerical test oracle.
    hidden = model(x)
    h_fused = hidden[:, :-1].detach().reshape(-1, 64).clone().requires_grad_()
    h_ref = h_fused.detach().clone().requires_grad_()
    ce_fused = fused(head.weight, h_fused, targets)
    ce_ref = F.cross_entropy(F.linear(h_ref, head.weight), targets, ignore_index=-100)
    torch.testing.assert_close(ce_fused, ce_ref, rtol=2e-4, atol=2e-5)
    ce_fused.backward()
    ce_ref.backward()
    torch.testing.assert_close(h_fused.grad, h_ref.grad, rtol=2e-3, atol=2e-5)
    print('PASS: fused CE value and hidden-state gradients match reference CE', flush=True)

    model.zero_grad(set_to_none=True)
    giou = coordinate_giou_loss(model(x), head, ids, **meta)
    giou.backward()  # No CE: proves an independent gradient path into real PEFT.
    nonzero = []
    for name, p in trainable:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        norm = p.grad.norm().item()
        print(f'GIoU-only {name}: grad_norm={norm:.8g}', flush=True)
        if norm > 0:
            nonzero.append(name)
    # Standard LoRA B starts at zero; A may legitimately have zero gradient here.
    assert any('lora_B' in n for n in nonzero), 'GIoU did not reach LoRA B'
    print('PASS: GIoU reaches real PEFT adapters through a frozen head', flush=True)

    def losses():
        hidden = model(x)
        ce = fused(head.weight, hidden[:, :-1].contiguous().reshape(-1, 64), targets)
        box = coordinate_giou_loss(hidden, head, ids, **meta)
        return ce, box, ce + 0.1 * box

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.003, weight_decay=0)
    with torch.no_grad():
        initial_total = losses()[2].item()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ce, box, total = losses()
        assert torch.isfinite(torch.stack([ce, box, total])).all()
        torch.testing.assert_close(ce + 0.0 * box, ce, rtol=0, atol=0)
        total.backward()
        for name, p in trainable:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f'step={step} ce={ce.item():.6f} giou={box.item():.6f} '
                  f'total={total.item():.6f} boxes=2', flush=True)

    with torch.no_grad():
        final_total = losses()[2].item()
    assert any(not torch.equal(initial[n], p) for n, p in trainable)
    assert head.weight.grad is None
    torch.testing.assert_close(head.weight, head_initial, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert final_total < initial_total, (initial_total, final_total)
    print(f'PASS: combined objective decreased {initial_total:.6f} -> {final_total:.6f}')
    print(f'Peak CUDA allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB')
    print('ALL CHECKS PASSED (FP32; no full-model/MagiAttention/distributed validation)')


if __name__ == '__main__':
    main()
