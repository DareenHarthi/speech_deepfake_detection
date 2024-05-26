
import argparse
from models.embeddings import process_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio embeddings based on the specified type.")
    parser.add_argument("folder", type=str, help="The folder path containing WAV files.")
    parser.add_argument("embedding_type", type=str, help="The type of embedding to process (pengi, clap, wavlm, wav2vec).")
    args = parser.parse_args()

    process_embeddings(args.folder, args.embedding_type)