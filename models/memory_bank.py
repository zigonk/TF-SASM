import torch
import torchvision
from torch import nn
from models.structures import Instances

from util import box_ops


class MemoryBankBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self.max_his_length = args.memory_bank_len
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, track_instances: Instances) -> Instances:
        raise NotImplementedError()

    def _update_memory_bank(self, track_instances):
        raise NotImplementedError()


class SparsenessAwareMemoryModule(MemoryBankBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        # Scale to 0-1 relative to image size
        self.max_accum_dist = args.memory_bank_max_dist
        self.max_frame_gap = args.memory_bank_max_frame_gap
        self.iou_threshold = args.memory_bank_iou_threshold

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        pass

    def _select_active_tracks(self, track_instances: Instances) -> Instances:
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (
                track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_memory_bank(self, track_instances):
        """Update memory bank based on accumulated moving distance of objects. 
        Args:
            track_instances (Instances): Instances of tracks

        Returns:
            Instances: Updated instances of tracks
        """
        if len(track_instances) == 0:
            return track_instances

        frame_gaps = track_instances.frame_idx - track_instances.mem_frames_idx[:, -1]
        accum_dists = track_instances.accum_dist
        is_update = (frame_gaps > self.max_frame_gap) | (
            accum_dists > self.max_accum_dist)
        
        pred_boxes = track_instances.pred_boxes.detach().clone()
        # Convert to xyxy format
        pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
        # Create a matrix masked at (i,j) if y2_j < y2_i
        mask_matrix = pred_boxes[:, 3].unsqueeze(0) < pred_boxes[:, 3].unsqueeze(1)

        # Set the masked matrix to 0
        ious = torchvision.ops.box_iou(pred_boxes, pred_boxes)
        # Remove self iou
        ious = ious * mask_matrix
        ious[range(len(ious)), range(len(ious))] = 0
        # Masked ious with mask_matrix
        ious = ious.max(dim=-1).values
        is_occluded = ious > self.iou_threshold

        is_update = is_update & (~is_occluded)
        # print(f"Update {is_update.sum()} tracks")

        is_pos = track_instances.scores > 0.8
        is_update = is_update & is_pos

        track_instances.accum_dist[is_update] = torch.zeros_like(
            track_instances.accum_dist[is_update])
        
        new_mem_frames_idx = track_instances.mem_frames_idx[is_update].clone()
        new_mem_bank = track_instances.mem_bank[is_update].clone()
        new_mem_padding_mask = track_instances.mem_padding_mask[is_update].clone()
        # Shifted all mem_bank and mem_padding_mask of updated idx to left
        new_mem_frames_idx[:, :-1] = track_instances.mem_frames_idx[is_update][:, 1:]
        new_mem_bank[:, :-1] = track_instances.mem_bank[is_update][:, 1:]
        new_mem_padding_mask[:, :-1] = track_instances.mem_padding_mask[is_update][:, 1:]
        # Update the last column of mem_bank and mem_padding_mask
        new_mem_frames_idx[:, -1] = track_instances.frame_idx[is_update]
        new_mem_bank[:, -1] = track_instances.query_pos[is_update]
        new_mem_padding_mask[:, -1] = torch.zeros_like(new_mem_padding_mask[:, -1])
        
        track_instances.mem_frames_idx[is_update] = new_mem_frames_idx
        track_instances.mem_bank[is_update] = new_mem_bank
        track_instances.mem_padding_mask[is_update] = new_mem_padding_mask

        return track_instances

    def forward(self, track_instances: Instances) -> Instances:
        # track_instances = self._select_active_tracks(track_instances)
        track_instances = self._update_memory_bank(track_instances)
        return track_instances


def build(args, dim_in, hidden_dim, dim_out):
    memory_banks = {
        'sam': SparsenessAwareMemoryModule,
    }
    return memory_banks[args.memory_bank_type](args, dim_in, hidden_dim, dim_out)
