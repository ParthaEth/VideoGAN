import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, PositionalEncodingPermute3D
import numpy as np
from transformers import LEDConfig, LEDModel


class PosEncGivenPos(torch.nn.Module):
    def __init__(self, proj_dim, enc_type: str):
        super().__init__()

        # import ipdb; ipdb.set_trace()
        if enc_type.lower() == 'fourier':
            std_pos_encoder = PositionalEncodingPermute3D(proj_dim)
            pos_enc = std_pos_encoder(torch.randn(1, proj_dim, 32, 32, 32))
            self.register_buffer('pos_enc', pos_enc)
        elif enc_type.lower() == 'learned':
            self.pos_enc = torch.nn.Parameter(torch.randn((1, proj_dim, 32, 32, 32)))
        else:
            raise NotImplementedError(f'Encoding type {enc_type} not implemented')

    def forward(self, coordinates):
        """coordinates: batch, num_pts, 3 of the format row, col, depth"""
        batch, n_pts, _ = coordinates.shape
        h_out = w_out = 1
        d_out = -1
        coordinates = coordinates.reshape(batch, d_out, h_out, w_out, 3)
        # since the grid sampler expects x,y,z format and row, col, depth maps to z, y, x we must shuffle
        coordinates = coordinates.flip(dims=(-1,))
        pe = torch.nn.functional.grid_sample(self.pos_enc.expand(batch, -1, -1, -1, -1), coordinates,
                                             align_corners=True,
                                             mode='bilinear',
                                             # mode='nearest',
                                             padding_mode='reflection')
        # now pe \in R^(batch, color_channels, d_out, h_out, w_out)
        return pe.reshape(batch, -1, n_pts).permute(0, 2, 1)


class LayerMhAttentionAndFeedForward(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(proj_dim, num_heads, dropout=0.1, bias=True, add_bias_kv=False,
                                               add_zero_attn=False, kdim=proj_dim, vdim=proj_dim,
                                               batch_first=True, device=None, dtype=None)
        self.layer_norm1 = torch.nn.LayerNorm(proj_dim)
        self.feed_forward = torch.nn.Sequential(torch.nn.Linear(proj_dim, proj_dim),
                                                torch.nn.ReLU(proj_dim),
                                                torch.nn.Linear(proj_dim, proj_dim))
        self.layer_norm2 = torch.nn.LayerNorm(proj_dim)

    def forward(self, query_pt, key, val):
        attn_output, attn_weight = self.mha(query=query_pt, key=key, value=val, need_weights=True,
                                            average_attn_weights=True)
        attn_output = self.layer_norm1(attn_output + query_pt)
        attn_output = attn_output + self.feed_forward(attn_output)
        return self.layer_norm2(attn_output), attn_weight

class MHprojector(torch.nn.Module):
    def __init__(self, motion_feature_dim, appearance_feature_dim, num_heads):
        super().__init__()
        self.motion_feature_dim = motion_feature_dim
        self.appearance_feature_dim = appearance_feature_dim

        self.planes_pos_encoder = PositionalEncodingPermute2D(motion_feature_dim)
        self.pix_loc_pe = PosEncGivenPos(motion_feature_dim, enc_type='fourier')

        self.motion_appearance_query_x_attention = torch.nn.MultiheadAttention(motion_feature_dim, num_heads,
                                                                               dropout=0.1,  bias=True,
                                                                               add_bias_kv=False, add_zero_attn=False,
                                                                               kdim=2*motion_feature_dim,
                                                                               vdim=motion_feature_dim,
                                                                               batch_first=True, device=None,
                                                                               dtype=None)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=2*motion_feature_dim, dim_feedforward=2048, nhead=8,
                                                         batch_first=True)
        self.encode_movement = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.layer_norm1 = torch.nn.LayerNorm(motion_feature_dim)
        self.final_lin = torch.nn.Linear(motion_feature_dim, self.motion_feature_dim + self.appearance_feature_dim)
        self.appearance_with_time = None
        self.motion_latent = None

    def get_whole_video_pe(self, x_res, y_res, t_res, dtype, device):
        cod_x = torch.linspace(-1, 1, x_res, dtype=dtype, device=device)
        cod_y = torch.linspace(-1, 1, y_res, dtype=dtype, device=device)
        cod_z = torch.linspace(-1, 1, t_res, dtype=dtype, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(cod_x, cod_y, cod_z, indexing='ij')
        full_vid_coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
        full_vid_pix_pe = self.pix_loc_pe(full_vid_coordinates.reshape(1, -1, 3))
        return full_vid_pix_pe

    def forward(self, plane_features, query_pt, recompute_full_vid_features):
        """
            :param plane_features: (batch, ch, h, w).
            :param query_pt: (batch, n_points, 3).
        """
        batch, ch, h, w = plane_features.shape
        assert self.motion_feature_dim + self.appearance_feature_dim == ch
        motion_features = plane_features[:, :self.motion_feature_dim, :, :]
        motion_features_pe = self.planes_pos_encoder(motion_features).reshape(batch, self.motion_feature_dim, -1)
        motion_features = motion_features.reshape(batch, self.motion_feature_dim, -1)
        mf_and_pe = torch.cat((motion_features, motion_features_pe), dim=1).permute(0, 2, 1)
        pix_loc_pe = self.pix_loc_pe(query_pt)

        x_res, y_res, t_res = 16, 16, 16
        dtype, device = plane_features.dtype, plane_features.device
        if recompute_full_vid_features:
            self.motion_latent = torch.randn((batch, 1, self.motion_feature_dim), dtype=dtype, device=device)
            # self.motion_latent = self.motion_latent.expand(-1, x_res * y_res * t_res, -1)
            self.motion_latent = self.motion_latent.expand(-1, pix_loc_pe.shape[1], -1)

        full_vid_pix_loc_pe = self.get_whole_video_pe(x_res, y_res, t_res, dtype, device)
        full_vid_pix_loc_pe = full_vid_pix_loc_pe.expand(batch, -1, -1).repeat(1, 1, 2)
        #
        # full_vid_pix_loc_pe = torch.cat((full_vid_pix_loc_pe, self.full_vid_motion_features), dim=2)
        # self.appearance_with_time = self.encode_movement(memory=full_vid_pix_loc_pe, tgt=mf_and_pe)
        # pix_loc_pe_and_motion_latent = torch.cat((self.motion_latent, pix_loc_pe), dim=-1)
        # pix_loc_pe_and_motion_latent = torch.cat((pix_loc_pe, pix_loc_pe), dim=-1)
        appearance_with_time = self.encode_movement(memory=full_vid_pix_loc_pe, tgt=mf_and_pe)
        # import ipdb; ipdb.set_trace()

        appearance_features = plane_features[:, self.motion_feature_dim:, :, :]\
            .reshape(batch, self.appearance_feature_dim, -1).permute(0, 2, 1)
        attn_output, attn_mask = self.motion_appearance_query_x_attention(
            query=pix_loc_pe, key=appearance_with_time, value=appearance_features, need_weights=True)

        attn_output = self.final_lin(self.layer_norm1(attn_output))

        return attn_output, attn_mask # dim: batch, num_pts, features


class TransformerProjector(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        # self.model = torch.nn.Transformer(d_model=proj_dim, nhead=num_heads, batch_first=True, num_encoder_layers=2,
        #                                   num_decoder_layers=2, dim_feedforward=512,)

        configuration = LEDConfig()
        configuration.attention_window = 1024
        configuration.vocab_size = 1
        configuration.d_model = proj_dim
        configuration.decoder_start_token_id = 0
        configuration.pad_token_id = 0
        configuration.max_encoder_position_embeddings = 4096
        configuration.max_decoder_position_embeddings = 16384
        configuration.decoder_ffn_dim = configuration.encoder_ffn_dim = 1024
        configuration.decoder_attention_heads = configuration.encoder_attention_heads = num_heads
        configuration.decoder_layers = configuration.encoder_layers = 6

        self.model = LEDModel(configuration)

        self.planes_pos_encoder = PositionalEncodingPermute2D(proj_dim)
        self.q_map = PosEncGivenPos(proj_dim, enc_type='fourier')
        self.self_att_neighbourhood = 64
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

    def forward(self,  plane_features, query_pt, recompute_full_vid_features):
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
        transformed_out = self.model(inputs_embeds=plane_features, decoder_inputs_embeds=query_pt,
                                     return_dict=True)['last_hidden_state']
        # import ipdb; ipdb.set_trace()
        # return torch.cat(transformed_out, dim=split_axis)
        return transformed_out, torch.zeros((batch, h*w, num_pts), device=plane_features.device,
                                            dtype=plane_features.dtype)


class PositionaGenerator(torch.nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.std_pos_emb = PositionalEncodingPermute2D(proj_dim)
        self.make_generative_pos_emb = torch.nn.Sequential(torch.nn.Linear(2*proj_dim, 4*proj_dim),
                                                           torch.nn.ReLU(4*proj_dim),
                                                           torch.nn.Linear(4*proj_dim, 2*proj_dim),
                                                           torch.nn.ReLU(2 * proj_dim),
                                                           torch.nn.Linear(2*proj_dim, proj_dim))

    def forward(self, z, features):
        '''
        Z: vluees will be used
        features: Only positional embedding of the features wll be used
        '''
        batch, ch, h, w = features.shape
        fpe = self.std_pos_emb(features).permute(0, 2, 3, 1).reshape(batch, h * w, -1)
        # fpe is of shape batch, h*w, proj_dim
        z_proj_dim = z[:, :fpe.shape[-1]][:, None, :].expand(-1, h * w, -1)

        fpe_and_noise = torch.cat((fpe, z_proj_dim), dim=-1)
        generative_pos_emb = self.make_generative_pos_emb(fpe_and_noise)
        return generative_pos_emb + fpe


class TinyMHprojector(torch.nn.Module):
    def __init__(self, proj_dim, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        self.planes_pos_encoder = PositionaGenerator(proj_dim)

        self.q_map = PosEncGivenPos(proj_dim, enc_type='learned')

        self.attn_mod1 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        # self.attn_mod2 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        # self.attn_mod3 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)
        # self.attn_mod4 = LayerMhAttentionAndFeedForward(proj_dim, num_heads)

        self.final_lin = torch.nn.Linear(proj_dim, proj_dim)

    def forward(self, z, plane_features, query_pt, recompute_full_vid_features):
        """
            :param plane_features: (batch, ch, h, w).
            :param query_pt: (batch, n_points, 3).
            :param z: (batch, latent_dim); latent_dim >= proj_dim
        """
        assert self.proj_dim <= z.shape[-1]
        batch, ch, h, w = plane_features.shape
        assert self.proj_dim == ch
        keys = self.planes_pos_encoder(z, plane_features)
        values = plane_features.permute(0, 2, 3, 1).reshape(batch, h * w, self.proj_dim)

        # import ipdb; ipdb.set_trace()
        batch, num_pts, pt_dim = query_pt.shape
        assert pt_dim == 3

        query_pt = self.q_map(query_pt)
        attn_output, attn_mask = self.attn_mod1(query_pt, keys, values)
        # attn_output_interm = attn_output + query_pt
        # attn_output, _ = self.attn_mod2(attn_output_interm, attn_output_interm, attn_output_interm)
        # attn_output_interm = attn_output_interm + attn_output
        # attn_output, attn_mask = self.attn_mod3(attn_output_interm, keys, values)

        attn_output = self.final_lin(attn_output)

        return attn_output.reshape(batch, num_pts, -1), attn_mask # dim: batch, num_pts, features