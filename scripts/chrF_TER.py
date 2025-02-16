import sacrebleu
import pyter

# Function to read text files line by line
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Strip any extra whitespace/newlines
    return [line.strip() for line in lines]

# Function to calculate chrF2 score
def calculate_chrf2(reference_texts, hypothesis_texts):
    # Calculate chrF2 score (beta=2 for chrF2)
    chrf2_score = sacrebleu.corpus_chrf(hypothesis_texts, [reference_texts], beta=2)
    return chrf2_score.score

# Function to calculate TER score
def calculate_ter(reference_texts, hypothesis_texts):
    # Calculate TER score using pyter
    ter_scores = []
    for ref, hyp in zip(reference_texts, hypothesis_texts):
        ter_score = pyter.ter(ref.split(), hyp.split())
        ter_scores.append(ter_score)
    
    # Average TER score across the corpus
    return sum(ter_scores) / len(ter_scores)

# File paths for reference and hypothesis texts
reference_file = '/path/test.hin_Deva'  # Replace with your reference file path
hypothesis_file = '/path/generated_predictions.txt'  # Replace with your hypothesis file path

# Read the reference and hypothesis sentences from files
reference_texts = read_text_file(reference_file)
hypothesis_texts = read_text_file(hypothesis_file)

# Check if both files have the same number of lines
if len(reference_texts) != len(hypothesis_texts):
    raise ValueError("The number of sentences in the reference and hypothesis files do not match!")

# Calculate chrF2 score
chrf2 = calculate_chrf2(reference_texts, hypothesis_texts)
print(f"chrF2 Score: {chrf2:.2f}")

# Calculate TER score
ter = calculate_ter(reference_texts, hypothesis_texts)
print(f"TER Score: {ter:.2f}")

