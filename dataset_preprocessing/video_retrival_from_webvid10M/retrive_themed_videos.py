import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')


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
    matching_video_ids = batch_df.iloc[matched_indices]['videoid'].tolist()

    # Clear the index for next batch
    gpu_index.reset()

    return matching_video_ids


if __name__ == '__main__':
    # Load CSV in chunks
    chunk_size = 10  # Adjust the chunk size based on available memory
    match_threshold = 0.5  # Adjust the match threshold based on your requirements
    query = "flowers moving"
    query_embedding = model.encode([query])

    # Initialize FAISS GPU index
    dimension = model.get_sentence_embedding_dimension()
    cpu_index = faiss.IndexFlatL2(dimension)
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
    csv_path = '/is/cluster/scratch/pghosh/dataset/WebVid_10M'

    # Process the CSV in chunks
    for chunk in pd.read_csv('path/to/your/file.csv', chunksize=chunk_size):
        matching_video_ids = process_batch(chunk, query_embedding, gpu_index, match_threshold)
        print(matching_video_ids)

