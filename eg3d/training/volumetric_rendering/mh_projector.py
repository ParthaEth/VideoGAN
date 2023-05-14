import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, PositionalEncodingPermute3D
import numpy as np


class PosEncGivenPos(torch.nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        std_pos_encoder = PositionalEncodingPermute3D(proj_dim)
        pos_enc = std_pos_encoder(torch.zeros(1, proj_dim, 32, 32, 32))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, coordinates):
        batch, n_pts, _ = coordinates.shape
        h_out = w_out = 2
        d_out = -1
        coordinates = coordinates.reshape(batch, d_out, h_out, w_out, 3)
        pe = torch.nn.functional.grid_sample(self.pos_enc.expand(batch, -1, -1, -1, -1), coordinates,
                                             align_corners=False, mode='bilinear')
        return pe.reshape(batch, -1, n_pts).permute(0, 2, 1)

class LayerMhAttentionAndFeedForward(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(proj_dim, num_heads, dropout=0.1, bias=True, add_bias_kv=False,
                                               add_zero_attn=False, kdim=proj_dim + proj_dim,
                                               vdim=proj_dim + proj_dim,
                                               batch_first=True, device=None, dtype=None)
        self.layer_norm1 = torch.nn.LayerNorm(proj_dim)
        self.feed_forward = torch.nn.Sequential(torch.nn.Linear(proj_dim, proj_dim),
                                                torch.nn.ReLU(proj_dim),
                                                torch.nn.Linear(proj_dim, proj_dim))
        self.layer_norm2 = torch.nn.LayerNorm(proj_dim)

    def forward(self, query_pt, key, val):
        attn_output, _ = self.mha(query=query_pt, key=key, value=val)
        attn_output = self.layer_norm1(attn_output + query_pt)
        attn_output = attn_output + self.feed_forward(attn_output)
        return self.layer_norm2(attn_output)

class MHprojector(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        self.planes_pos_encoder = PositionalEncodingPermute2D(proj_dim)

        self.q_map = PosEncGivenPos(proj_dim)

        self.attn_mod1 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        self.attn_mod2 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        self.attn_mod3 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        self.attn_mod4 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)

        self.final_lin = torch.nn.Linear(proj_dim, proj_dim)

    def forward(self, plane_features, query_pt):
        """
            :param plane_features: (batch, ch, h, w).
            :param query_pt: (batch, n_points, 3).
        """
        batch, ch, h, w = plane_features.shape
        assert self.proj_dim == ch
        key_val_feature_planes = torch.cat((plane_features, self.planes_pos_encoder(plane_features)), dim=1)
        key_val_feature_planes = key_val_feature_planes.permute(0, 2, 3, 1)  # batch, h, w, ch

        key_val_feature_planes = key_val_feature_planes.reshape(batch, h * w, self.proj_dim + self.proj_dim)

        # import ipdb; ipdb.set_trace()
        batch, num_pts, pt_dim = query_pt.shape
        assert pt_dim == 3

        query_pt = self.q_map(query_pt)
        attn_output = self.attn_mod1(query_pt, key_val_feature_planes, key_val_feature_planes)
        attn_output = self.attn_mod2(attn_output, key_val_feature_planes, key_val_feature_planes)
        attn_output = self.attn_mod3(attn_output, key_val_feature_planes, key_val_feature_planes)
        attn_output = self.attn_mod4(attn_output, key_val_feature_planes, key_val_feature_planes)

        attn_output = self.final_lin(attn_output)

        return attn_output.reshape(batch, num_pts, -1) # dim: batch, num_pts, features


class TransformerProjector(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        self.model = torch.nn.Transformer(d_model=proj_dim, nhead=num_heads, batch_first=True, num_encoder_layers=2,
                                          num_decoder_layers=2, dim_feedforward=512,)
        self.planes_pos_encoder = PositionalEncodingPermute2D(proj_dim)
        self.q_map = PosEncGivenPos(proj_dim)
        self.self_att_neighbourhood = 32
        mask_for_img_pts = self.make_mask(64, 64, 1, self.self_att_neighbourhood)
        mask_for_vid_pts = self.make_mask(32, 32, 16, self.self_att_neighbourhood)
        self.target_masks = {64*64: mask_for_img_pts, 32*32*16: mask_for_vid_pts}

    def make_mask(self, rows, cols, time_steps, neighbourhood_to_attend):
        seq_len = np.prod([rows, cols, time_steps])
        mask_for_img_pts = torch.ones((seq_len, seq_len), dtype=torch.bool)
        mask_row = 0
        for row in range(rows):
            for col in range(cols):
                for time in range(time_steps):
                    mask_this_row = torch.ones((rows, cols, time_steps), dtype=torch.bool)
                    first_neighbour_row = max(0, row - neighbourhood_to_attend // 2)
                    first_neighbour_col = max(0, col - neighbourhood_to_attend // 2)
                    # first_neighbour_time = max(0, time - neighbourhood_to_attend // 2)

                    mask_this_row[first_neighbour_row:row + neighbourhood_to_attend // 2,
                                  first_neighbour_col:col + neighbourhood_to_attend // 2,
                                  # first_neighbour_time:time + neighbourhood_to_attend // 2, ] = False
                                  time] = False
                    mask_for_img_pts[mask_row, :] = mask_this_row.flatten()
                    mask_row += 1

        return mask_for_img_pts.squeeze()

    def forward(self,  plane_features, query_pt):
        batch, ch, h, w = plane_features.shape
        assert self.proj_dim == ch
        plane_features = plane_features.permute(0, 2, 3, 1)  # batch, h, w, ch
        plane_features = plane_features.reshape(batch, h * w, self.proj_dim)

        # import ipdb; ipdb.set_trace()
        batch, num_pts, pt_dim = query_pt.shape
        assert pt_dim == 3

        query_pt = self.q_map(query_pt)
        # import ipdb; ipdb.set_trace()
        # transformed_out = []
        # split_axis = 1
        # for plane_features_chunk, query_pt_chunk in zip(plane_features.split(1, dim=0), query_pt.split(1, dim=0)):
        #     transformed_out_chunk = self.model(plane_features_chunk, query_pt_chunk)
        #     transformed_out.append(transformed_out_chunk)
        # for query_pt_chunk in query_pt.split(query_pt.shape[1]//4, dim=split_axis):
        #     transformed_out_chunk = self.model(plane_features, query_pt_chunk)
        #     transformed_out.append(transformed_out_chunk)
        # transformed_out = self.model(plane_features, query_pt, tgt_mask=self.target_masks[num_pts].to(query_pt.device),)
        transformed_out = self.model(plane_features, query_pt, tgt_mask=None)
        # import ipdb; ipdb.set_trace()
        # return torch.cat(transformed_out, dim=split_axis)
        return transformed_out