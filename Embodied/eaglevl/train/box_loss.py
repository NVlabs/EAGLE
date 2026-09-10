"""Canonical box metadata and a coordinate-only, differentiable GIoU loss."""

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


def canonical_box_metadata(labels, coord_token_ids, box_start_id, box_end_id):
    """Extract exact supervised boxes BEFORE MTP expansion, in raw label positions.

    Token IDs are mapped by the explicit <0> ... <1000> vocabulary order.
    Ignored labels, partial boxes and coordinates outside markers cannot match.
    This runs independently per sample, so matches cannot span packed boundaries.
    """
    ids = torch.as_tensor(coord_token_ids, dtype=torch.long, device=labels.device)
    if labels.ndim != 1:
        raise ValueError("Expected one unexpanded label sequence")
    if labels.numel() < 6:
        return dict(bbox_coord_positions=labels.new_empty((0, 4)),
                    gt_boxes=torch.empty((0, 4), device=labels.device))
    windows = labels.unfold(0, 6, 1)
    # Sort only the lookup; values remain the indices in number_tokens_list.
    sorted_ids, values = ids.sort()
    targets = windows[:, 1:5]
    lookup = torch.searchsorted(sorted_ids, targets.contiguous()).clamp_max(ids.numel() - 1)
    is_coord = sorted_ids[lookup].eq(targets).all(-1)
    valid = windows[:, 0].eq(box_start_id) & windows[:, 5].eq(box_end_id) & is_coord
    boxes = values[lookup].float()
    # Invalid GT geometry is skipped, never silently rewritten.
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    starts = valid.nonzero(as_tuple=True)[0]
    return dict(bbox_coord_positions=starts[:, None] + torch.arange(1, 5, device=labels.device),
                gt_boxes=boxes[valid])


def coordinate_giou_loss(hidden_states, lm_head, coord_token_ids,
                         bbox_coord_positions, gt_boxes, chunk_boxes=128):
    """Use h[t-1] for target at t; project ONLY coordinate vocabulary rows.

    Packing's collator emits one sequence per forward. Positions refer to that
    concatenated sequence's labels, not its reset/repeated position_ids.
    Works with a frozen/tied head: gradients still flow back into hidden states.
    """
    if hidden_states.shape[0] != 1:
        raise ValueError("Box metadata expects the stream packing batch size of one")
    if bbox_coord_positions.ndim != 2 or bbox_coord_positions.shape[1] != 4:
        raise ValueError("bbox_coord_positions must have shape [num_boxes, 4]")
    if gt_boxes.shape != bbox_coord_positions.shape:
        raise ValueError("gt_boxes must match bbox_coord_positions")
    if bbox_coord_positions.numel() == 0:
        return hidden_states.reshape(-1)[:0].float().sum()
    if torch.any(bbox_coord_positions < 1) or torch.any(bbox_coord_positions >= hidden_states.shape[1]):
        raise ValueError("Box target positions are outside the shifted sequence")
    ids = torch.as_tensor(coord_token_ids, dtype=torch.long, device=hidden_states.device)
    weight = lm_head.weight.index_select(0, ids)
    bias = getattr(lm_head, "bias", None)
    bias = bias.index_select(0, ids) if bias is not None else None
    coord_values = torch.arange(ids.numel(), device=hidden_states.device, dtype=torch.float32)
    scale = ids.numel() - 1  # verified ordered vocabulary: <0> ... <1000>
    total = hidden_states.reshape(-1)[:0].float().sum()
    for start in range(0, bbox_coord_positions.shape[0], chunk_boxes):
        positions = bbox_coord_positions[start:start + chunk_boxes]
        coord_hidden = hidden_states[0].index_select(0, (positions - 1).reshape(-1))
        logits = F.linear(coord_hidden, weight, bias)
        probabilities = torch.softmax(logits.float(), dim=-1)
        raw = (probabilities * coord_values).sum(-1).reshape(-1, 4) / scale
        pred = torch.cat((torch.minimum(raw[:, :2], raw[:, 2:]),
                          torch.maximum(raw[:, :2], raw[:, 2:])), dim=-1)
        target = gt_boxes[start:start + chunk_boxes].float() / scale
        total = total + generalized_box_iou_loss(pred, target, reduction="sum", eps=1e-7)
    return total / bbox_coord_positions.shape[0]
