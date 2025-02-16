import json
from comet import download_model, load_from_checkpoint

# Function to read text files line by line
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# File paths for reference, hypothesis, and source (if available)
source_file = '/path/test.mai_Deva'  # Maithili source sentences
reference_file = '/path/test.hin_Deva'  # Hindi reference translations
hypothesis_file = '/path/mai_hin_output.txt'  # Model predictions

# Read data from files
source_texts = read_text_file(source_file)
reference_texts = read_text_file(reference_file)
hypothesis_texts = read_text_file(hypothesis_file)

# Ensure all files have the same number of lines
if len(source_texts) != len(reference_texts) or len(reference_texts) != len(hypothesis_texts):
    raise ValueError("Mismatch: The number of sentences in source, reference, and hypothesis files do not match!")

# Load the COMET model
model_path = download_model("Unbabel/wmt22-comet-da")  # You can replace with other models like wmt20-comet-da
model = load_from_checkpoint(model_path)

# Prepare input data for COMET
data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(source_texts, hypothesis_texts, reference_texts)]

# Compute COMET scores
comet_output = model.predict(data, batch_size=8)

# Print the overall COMET Score
print(f"COMET Score: {comet_output['system_score']:.4f}")

# Optional: Print sentence-level COMET scores
for idx, score in enumerate(comet_output["scores"][:5]):  # Print first 5 examples
    print(f"Sentence {idx+1}: COMET Score = {score:.4f}")

