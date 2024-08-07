
import argparse
from models.embeddings import get_clap_embed_file
from models.model import MLPClassifier
import torch 


model = MLPClassifier(input_dim=1024)
model.load_state_dict(torch.load("checkpoints/clap_classifier.pth"))
 
def detect(file):
    
    embed = get_clap_embed_file(file)
    score = model(embed)
    return score.item()
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio embeddings based on the specified type.")
    parser.add_argument("file", type=str, help="The folder path containing WAV files.")
    args = parser.parse_args()
    
    score = detect(args.file)
    
    if score > 0.6:
        print(f"Audio is generated, with a confidence score of {score*100:.2f}%.")
    else:
        score = 1 - score
        print(f"Audio is likely real, with a confidence score of {score*100:.2f}%.")
        


    