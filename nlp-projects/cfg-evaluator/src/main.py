import sys
from pathlib import Path
from nltk import CFG
from nltk.parse.chart import ChartParser

# Source: https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/
# Source: https://www.nltk.org/howto/parse.html
def parse_dataset(input_file, output_file, grammar):
    TP, FP, TN, FN = 0, 0, 0, 0
    parser = ChartParser(grammar)
    
    with open(output_file, "w") as f:
        f.write("id\tground_truth\tprediction\n")
    
    header = True
    with open(input_file, 'r') as f:
        for line in f:
            # Skip writing the header from the input file
            if header:
                header = False
                continue  
            
            l = line.strip().split('\t')
            # Get the POS tags from the input file (e.g. ['NN', 'VBZ', 'DT', 'NN', '.'])
            pos = l[3].split()
            
            # Attempt to parse the POS tags
            # If the tags adhere to the grammar, the prediction is 0
            # Source: https://stackoverflow.com/questions/42966067/nltk-chart-parser-is-not-printing
            tree = list(parser.parse(pos))
            if tree:
                pred = 0
            # If the tags do not adhere to the grammar, the prediction is 1
            else:
                pred = 1
                                
            # Classify prediction as TP, FP, TN, or FN
            if l[1] == '1' and pred == 1:
                TP += 1
            elif l[1] == '0' and pred == 1:
                FP += 1
            elif l[1] == '0' and pred == 0:
                TN += 1
            elif l[1] == '1' and pred == 0:
                FN += 1
           
            with open(output_file, "a") as f:
                f.write(f"{l[0]}\t{l[1]}\t{pred}\n")
                
    precision_recall(TP, FP, TN, FN)

# Source: https://www.nltk.org/howto/grammar.html
def read_grammar(grammar_file):
    with open(grammar_file, 'r') as file:
        grammar_text = file.read()
    return CFG.fromstring(grammar_text)

# Displays confusion matrix, precision, and recall
def precision_recall(TP, FP, TN, FN):
    print("                | ground_truth = 1    | ground_truth = 0     |")
    print("--------------------------------------------------------------")
    print(f"prediction = 1  |       TP = {TP}      |       FP = {FP}       |")
    print(f"prediction = 0  |       FN = {FN}       |       TN = {TN}       |")
    print("--------------------------------------------------------------")
    
    # Round precision and recall to 4 decimal places, simple error handling for division by zero
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
        
    if TP == 0 and FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
            
def main():
    # Example usage: python3 src/main.py data/train.tsv grammars/toy.cfg output/train.tsv
    input_file = Path(sys.argv[1])
    grammar_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    grammar = read_grammar(grammar_file)
    parse_dataset(input_file, output_file, grammar)

if __name__ == '__main__':
    main()