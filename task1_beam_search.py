import numpy as np
from math import log

class NMTModel:
    def __init__(self):
        self.size_of_vocab = 5
        self.hidden_of_size = 10
        self.output_of_size = 5
        self.weights = {
            'encoder': np.random.randn(self.hidden_of_size, self.size_of_vocab),
            'decoder': np.random.randn(self.output_of_size, self.hidden_of_size)
        }

    def translate(self, sequence_of_input):
        output_encoded = np.dot(self.weights['encoder'], sequence_of_input)
        decoding_the_output = np.dot(self.weights['decoder'], output_encoded)
        return decoding_the_output

def beam_search_decoder_nmt(model, sequence_of_input, k):
    code_sequence = [[[], 0.0]]  
    for step_into_input in sequence_of_input:
        all_candidates = []

        for seq, score in code_sequence:
            for j in range(model.output_of_size):
                candidate_seq = seq + [j]
                decoding_the_output = model.translate(step_into_input)
                probability = decoding_the_output[j]
                if probability > 0:
                    candidate_score = score - log(probability)
                    all_candidates.append((candidate_seq, candidate_score))
 
        ordered = sorted(all_candidates, key=lambda x: x[1])
        code_sequence = ordered[:k]
    
    return code_sequence

def getting_the_user_input_sequence(size_of_vocab):
    sequence = []
    while True:
        input_str = input(f"Enter probabilities for {size_of_vocab} elements separated by space (e.g., 0.1 0.2 0.3 0.4 0.5): \n")
        probabilities = input_str.split()
        if len(probabilities) == size_of_vocab:
            try:
                sequence.append([float(prob) for prob in probabilities])
                break
            except ValueError:
                print("Please enter valid probabilities.")
        else:
            print(f"Please enter {size_of_vocab} probabilities.")
    return np.array(sequence)

nmt_model = NMTModel()

size_of_vocab = nmt_model.size_of_vocab
user_input_sequence = getting_the_user_input_sequence(size_of_vocab)

width_of_beam = 3
result = beam_search_decoder_nmt(nmt_model, user_input_sequence, width_of_beam)

for seq, score in result:
    print("Sequence:", seq, "Score:", score)