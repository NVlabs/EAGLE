"""CPU checks; run: python -m unittest discover -s tests -p test_box_loss.py -v.

Extract production dataset/packing methods via AST to avoid importing CUDA-only
Magi/Liger modules. Their method bodies execute unchanged, with a toy tokenizer.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torchvision.ops import generalized_box_iou_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eaglevl.train.box_loss import canonical_box_metadata, coordinate_giou_loss


def production_function(path, name, scope):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), scope)
    return scope[name]


class Tokenizer:
    pad_token_id = 0
    special = {'<|im_start|>': 1, 'assistant': 2, '<|im_end|>': 4,
               '<text_mask>': 5, '<null>': 6, '<box>': 7, '</box>': 8, '</ref>': 9}

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        return self.special[token] if token in self.special else 100 + 2 * int(token[1:-1])


class BoxLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.ids = list(range(100, 2101, 2))  # arbitrary, NON-contiguous vocabulary IDs
        self.scope = dict(torch=torch, np=np, copy=copy, Dict=Dict, List=List,
                          Optional=Optional, StreamPackedDatasetMTP=object,
                          IGNORE_TOKEN_ID=-100, BOX_START_TOKEN='<box>', BOX_END_TOKEN='</box>',
                          number_tokens_list=[f'<{i}>' for i in range(1001)],
                          canonical_box_metadata=canonical_box_metadata)
        self.source = ROOT / 'eaglevl/train/locany_finetune_magi_stream.py'

    def sample(self):
        box = [7, self.ids[100], self.ids[200], self.ids[700], self.ids[800], 8]
        # A box in the user message MUST NOT be supervised or extracted.
        tokens = torch.tensor([1, 10, 3] + box + [4, 1, 2, 3] + box + box + [4])
        dataset = SimpleNamespace(processor=SimpleNamespace(tokenizer=Tokenizer()),
                                  block_size=6, ds_name='test')
        fn = production_function(self.source, 'get_targets_flag_with_mtp', self.scope)
        result = fn(dataset, tokens)
        result.update(pixel_values=torch.zeros(4, 8), image_flags=torch.tensor([0]),
                      image_grid_hws=np.array([[2, 2]]))
        return tokens, result

    def test_canonical_mtp_and_packing_alignment(self):
        original, sample = self.sample()
        self.assertEqual(sample['bbox_coord_positions'].shape, (2, 4))
        self.assertGreater(sample['input_ids'].numel(), original.numel())
        self.assertTrue(torch.all(sample['bbox_coord_positions'] < original.numel()))
        self.assertTrue(torch.any(sample['labels'][original.numel():] >= 100))
        merge = production_function(self.source, '_merge_samples', self.scope)
        finalize = production_function(self.source, '_finalize_batch', self.scope)
        collate = production_function(self.source, 'packed_collate_fn_mtp', self.scope)
        first = merge(None, None, sample)
        packed = merge(None, first, sample)
        torch.testing.assert_close(packed['bbox_coord_positions'][2:],
                                   sample['bbox_coord_positions'] + len(sample['input_ids']))
        packed = finalize(None, packed)
        batch = collate([packed])
        positions = batch['bbox_coord_positions']
        expected_ids = 100 + 2 * batch['gt_boxes'].long()
        torch.testing.assert_close(batch['labels'][0, positions], expected_ids)
        # The actual CE-shift target indexed with position-1 must be identical.
        torch.testing.assert_close(batch['labels'][:, 1:][0, positions - 1], expected_ids)
        packed['bbox_coord_positions'][0] = torch.arange(len(sample['input_ids']) - 2,
                                                         len(sample['input_ids']) + 2)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            collate([packed])

    def test_invalid_ignored_and_nonbox_coordinates(self):
        c = self.ids
        labels = torch.tensor([c[10], c[20], c[30], c[40],
                               7, c[10], c[20], -100, c[40], 8,
                               7, c[10], c[20], c[30], 8,
                               7, c[30], c[20], c[10], c[40], 8,
                               7, c[10], c[20], c[30], c[40]])
        meta = canonical_box_metadata(labels, c, 7, 8)
        self.assertEqual(meta['gt_boxes'].shape, (0, 4))

    def test_giou_matches_geometry(self):
        target = torch.tensor([[0., 0., 1., 1.]])
        torch.testing.assert_close(generalized_box_iou_loss(target, target), torch.zeros(1), atol=2e-7, rtol=0)
        torch.testing.assert_close(generalized_box_iou_loss(torch.tensor([[2., 0., 3., 1.]]), target),
                                   torch.tensor([4 / 3]))
        self.assertTrue(torch.isfinite(generalized_box_iou_loss(torch.zeros(1, 4), target)).all())

    def test_unsorted_token_ids_and_tied_head(self):
        ids = self.ids[::-1]
        labels = torch.tensor([-100, 7, ids[100], ids[200], ids[700], ids[800], 8])
        meta = canonical_box_metadata(labels, ids, 7, 8)
        torch.testing.assert_close(meta['gt_boxes'], torch.tensor([[100., 200., 700., 800.]]))
        embedding = nn.Embedding(2200, 8)
        head = nn.Linear(8, 2200, bias=False)
        head.weight = embedding.weight
        hidden = torch.randn(1, 7, 8, requires_grad=True)
        loss = coordinate_giou_loss(hidden, head, ids, **meta)
        loss.backward()
        self.assertIs(head.weight, embedding.weight)
        self.assertGreater(embedding.weight.grad.abs().sum(), 0)
        self.assertGreater(hidden.grad.abs().sum(), 0)

    def test_shift_restricted_projection_bias_and_head_gradients(self):
        hidden = torch.randn(1, 12, 8, requires_grad=True)
        head = nn.Linear(8, 2200, bias=True)
        positions = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 10]])
        gt = torch.tensor([[100., 200., 700., 800.], [50., 80., 950., 980.]])
        loss = coordinate_giou_loss(hidden, head, self.ids, positions, gt, chunk_boxes=1)
        logits = head(hidden[0, (positions - 1).reshape(-1)])[:, self.ids].float()
        raw = (logits.softmax(-1) * torch.arange(1001)).sum(-1).reshape(-1, 4) / 1000
        pred = torch.cat([torch.minimum(raw[:, :2], raw[:, 2:]),
                          torch.maximum(raw[:, :2], raw[:, 2:])], -1)
        torch.testing.assert_close(loss, generalized_box_iou_loss(pred, gt / 1000, reduction="mean"))
        loss.backward()
        self.assertGreater(head.weight.grad.abs().sum(), 0)
        self.assertGreater(head.bias.grad.abs().sum(), 0)
        self.assertEqual(head.weight.grad[101].abs().sum(), 0)  # not a coordinate row
        expected = torch.zeros(12, dtype=torch.bool)
        expected[(positions - 1).reshape(-1)] = True
        torch.testing.assert_close(hidden.grad[0].abs().sum(-1) > 0, expected)

    def test_frozen_head_reaches_low_rank_adapters(self):
        base = nn.Linear(8, 8).requires_grad_(False)
        lora_A = nn.Linear(8, 2, bias=False)
        lora_B = nn.Linear(2, 8, bias=False)
        head = nn.Linear(8, 2200, bias=False).requires_grad_(False)
        inputs = torch.randn(1, 8, 8)
        hidden = base(inputs) + lora_B(lora_A(inputs))
        loss = coordinate_giou_loss(hidden, head, self.ids, torch.tensor([[2, 3, 4, 5]]),
                                    torch.tensor([[100., 200., 700., 800.]]))
        loss.backward()  # GIoU ALONE, no CE contribution
        for adapter in (lora_A, lora_B):
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().sum(), 0)
        self.assertIsNone(head.weight.grad)
        print(f'GIoU-only adapter gradients: A={lora_A.weight.grad.norm():.6g}, '
              f'B={lora_B.weight.grad.norm():.6g}')

    def test_empty_boxes_and_bfloat16(self):
        hidden = torch.randn(1, 8, 8, requires_grad=True)
        head = nn.Linear(8, 2200)
        zero = coordinate_giou_loss(hidden, head, self.ids, torch.empty(0, 4, dtype=torch.long),
                                    torch.empty(0, 4))
        self.assertEqual(zero, 0)
        zero.backward()
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = coordinate_giou_loss(hidden, head, self.ids, torch.tensor([[2, 3, 4, 5]]),
                                        torch.tensor([[100., 200., 700., 800.]]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(hidden.grad.abs().sum(), 0)

    def test_production_forward_preserves_ce_and_adds_giou(self):
        _, sample = self.sample()
        captured = {}

        class ReferenceCE(nn.Module):
            # CPU reference substitutes ONLY the unavailable CUDA fused kernel.
            def __init__(self, ignore_index, reduction):
                super().__init__()

            def forward(self, weight, hidden, labels):
                captured['labels'] = labels.clone()
                captured['ce'] = torch.nn.functional.cross_entropy(hidden @ weight.T, labels)
                return captured['ce']

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(8, 8).requires_grad_(False)
                self.lora_A = nn.Linear(8, 2, bias=False)
                self.lora_B = nn.Linear(2, 8, bias=False)

            def forward(self, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=self.base(inputs_embeds)
                                       + self.lora_B(self.lora_A(inputs_embeds)),
                                       past_key_values=None, hidden_states=None, attentions=None)

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(2200, 8).requires_grad_(False)
                self.lm_head = nn.Linear(8, 2200, bias=False).requires_grad_(False)
                self.model = Decoder()

            def get_input_embeddings(self):
                return self.embedding

        scope = dict(torch=torch, List=List, Optional=Optional, Union=Union, Tuple=Tuple,
                     CausalLMOutputWithPast=SimpleNamespace, IGNORE_INDEX=-100,
                     LigerFusedLinearCrossEntropyLoss=ReferenceCE,
                     coordinate_giou_loss=coordinate_giou_loss)
        forward = production_function(ROOT / 'eaglevl/model/locany/modeling_locateanything.py',
                                      'forward', scope)

        class Model(nn.Module):
            def __init__(self, ids):
                super().__init__()
                self.config = SimpleNamespace(use_return_dict=True, lambda_box=0.1, coord_token_ids=ids)
                self.language_model = Language()
                self.mlp1 = nn.Identity()
                self.use_llm_lora = False  # CPU decoder contains the trainable low-rank layers

            def extract_feature(self, pixels, grid):
                return [torch.zeros(1, 8)]

        Model.forward = forward
        model = Model(self.ids)
        kwargs = dict(pixel_values=sample['pixel_values'], input_ids=sample['input_ids'][None],
                      labels=sample['labels'][None], image_flags=torch.tensor([0]),
                      sub_sample_lengths=[torch.tensor([len(sample['input_ids'])])],
                      bbox_coord_positions=sample['bbox_coord_positions'], gt_boxes=sample['gt_boxes'])
        result = model(**kwargs)
        ce, giou, total, count = model._last_box_loss_metrics
        torch.testing.assert_close(result.loss, captured['ce'] + 0.1 * giou)
        torch.testing.assert_close(captured['labels'], sample['labels'][1:])
        self.assertEqual(count, 2)
        result.loss.backward()
        self.assertGreater(model.language_model.model.lora_B.weight.grad.abs().sum(), 0)
        model.config.lambda_box = 0
        result = model(**kwargs)
        torch.testing.assert_close(result.loss, captured['ce'], rtol=0, atol=0)
        self.assertIsNone(result.logits)
        model.config.lambda_box = 0.1
        kwargs.update(bbox_coord_positions=torch.empty(0, 4, dtype=torch.long),
                      gt_boxes=torch.empty(0, 4))
        result = model(**kwargs)
        torch.testing.assert_close(result.loss, captured['ce'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
