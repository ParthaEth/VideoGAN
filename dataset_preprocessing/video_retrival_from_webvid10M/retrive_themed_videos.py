import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tqdm


# Define a function to process a batch
def process_batch(batch_df, query_embedding, gpu_index, match_threshold):
    # Encode the video names
    document_embeddings = model.encode(batch_df['name'].tolist())

    # Add to FAISS GPU index
    gpu_index.add(np.array(document_embeddings))

    # Search the index
    D, I = gpu_index.search(query_embedding, len(batch_df))

    # Filter results based on match threshold
    matched_indices = I[0][D[0] < match_threshold]
    matching_video_ids = batch_df.iloc[matched_indices]['videoid'].tolist(), \
        batch_df.iloc[matched_indices]['name'].tolist()

    # Clear the index for next batch
    gpu_index.reset()

    return matching_video_ids


if __name__ == '__main__':
    # Load CSV in chunks
    chunk_size = 1000  # Adjust the chunk size based on available memory
    match_threshold = 0.9  # Adjust the match threshold based on your requirements
    query = "flowers moving in the wind"
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    query_embedding = model.encode([query])

    # Initialize FAISS GPU index
    dimension = model.get_sentence_embedding_dimension()
    cpu_index = faiss.IndexFlatL2(dimension)
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
    # csv_path = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/results_10M_val.csv'
    csv_path = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/results_10M_train.csv'
    output_csv_path = 'flowers_train_video_ids.csv'
    total_rows = sum(1 for _ in open(csv_path)) - 1

    # Process the CSV in chunks
    with open(output_csv_path, 'w') as f:
        for chunk in tqdm.tqdm(pd.read_csv(csv_path, chunksize=chunk_size), total=total_rows // chunk_size + 1):
            # import ipdb; ipdb.set_trace()
            matching_video_ids = process_batch(chunk, query_embedding, gpu_index, match_threshold)
            # if matching_video_ids[-1]:
                # print(matching_video_ids[-1])

            # Append the matching video IDs and names to the output CSV file
            for videoid in matching_video_ids[0]:
                # import ipdb; ipdb.set_trace()
                f.write(f'{videoid}\n')
