from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from transformers import LEDConfig, LEDModel

import torch
from torch.optim import Adam


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


if __name__ == '__main__':
    feature_dim = 96
    batch_size = 4
    feature_res = 64
    time_steps = 16
    device = 'cuda'

    configuration = LEDConfig()
    configuration.attention_window = 1024
    configuration.vocab_size = 1
    configuration.d_model = feature_dim
    configuration.decoder_start_token_id = 0
    configuration.pad_token_id = 0
    configuration.max_encoder_position_embeddings = 4096
    configuration.max_decoder_position_embeddings = 16384
    configuration.decoder_ffn_dim = configuration.encoder_ffn_dim = 1024
    configuration.decoder_attention_heads = configuration.encoder_attention_heads = 1
    configuration.decoder_layers = configuration.encoder_layers = 6


    model = LEDModel(configuration).to(device)

    # model = torch.nn.Transformer(d_model=feature_dim, nhead=1, batch_first=True, num_encoder_layers=2,
    #                                           num_decoder_layers=2, dim_feedforward=512,).to(device)
    print(f'Memory to hold the initialized transformer model')
    print_gpu_utilization()
    print('-------------------------------------------------')

    print(f'Memory to hold the optimizer of the transformer model')
    optimizer = Adam(model.parameters(), lr=0.1)
    print_gpu_utilization()
    print('-------------------------------------------------')

    plane_features = torch.randn(batch_size, feature_res*feature_res, feature_dim).to(device)
    query_pt = torch.randn(batch_size, 32*32*time_steps, feature_dim).to(device)
    dest = torch.randn(batch_size, 32*32*time_steps, feature_dim).to(device)

    tgt_mask = torch.ones((32*32*time_steps, 32*32*time_steps), dtype=torch.bool, device=device)
    tgt_mask[:, :64] = False
    tgt_mask = None

    for i in range(1000):
        print(f'Memory to make a forward pass with shape {plane_features.shape} and query shape {query_pt.shape} '
              f'transformer model')
        # transformed_out = model(plane_features, query_pt, tgt_mask=tgt_mask)
        transformed_out = model(inputs_embeds=plane_features, decoder_inputs_embeds=query_pt,
                                return_dict=True)['last_hidden_state']

        print_gpu_utilization()
        print('-------------------------------------------------')

        error = (dest - transformed_out).mean()
        print(f'Memory to make a backward pass with shape {plane_features.shape} transformer model')
        error.backward()
        optimizer.step()
        print_gpu_utilization()
        print('-------------------------------------------------')


