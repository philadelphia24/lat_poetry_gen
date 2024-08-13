###############################################################################
# Language Modeling 
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch
import data
import json
import re
from cltk.prosody.lat.hexameter_scanner import HexameterScanner

parser = argparse.ArgumentParser(description='PyTorch Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/corpus_10mio_10k', # CHANGE
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/model_10mio_10k.pt', # CHANGE
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='./generated/ten_million/generated_10mio_10k_cltk.txt', # CHANGE
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--dict', type=str, default='./syllable_dictionaries/syllable_dict_10mio_10k.json', help='path to syllable dictionary') # CHANGE
args = parser.parse_args()


# Load syllable dictionary
with open(args.dict, 'r', encoding='utf-8') as f:
    syllable_dict = json.load(f)

# Define punctuation marks
punctuation_marks = {'.', ',', ';', ':', '!', '?'}

# CLTK scanner
scanner = HexameterScanner()

# Dactylic hexameter structure
dactylic_hexameter = [
    ["lss", "ll"],  # First foot: dactyl or spondee
    ["lss", "ll"],  # Second foot: dactyl or spondee
    ["lss", "ll"],  # Third foot: dactyl or spondee
    ["lss", "ll"],  # Fourth foot: dactyl or spondee
    ["lss"],  # Fifth foot: usually dactyl
    ["ll", "ls"]  # Sixth foot: spondee or anceps
]


def syllable_to_foot(syllable_pattern):
    # Flexibly interpret 'u' as either 'l' or 's'
    pattern_flex = syllable_pattern.replace('u', '[ls]')
    return pattern_flex

def fits_foot(word_syl, foot_pattern):
    word_foot_flex = syllable_to_foot(word_syl)
    # ensure the whole string matches the pattern exactly
    return any(re.match(f"^{word_foot_flex}$", p) for p in foot_pattern)

def is_punctuation(word):
    return word in punctuation_marks

def remove_breves(text):
    return text.replace('ă', 'a').replace('ĕ', 'e').replace('ĭ', 'i').replace('ŏ', 'o').replace('ŭ', 'u')

def adjust_prob_distribution(output_probs, current_foot_idx, dactylic_hexameter, syllable_dict, last_word, last_is_punctuation):
    current_foot = dactylic_hexameter[current_foot_idx]
    valid_indices = []

    for word, info in syllable_dict.items():
        if fits_foot(info['syl'], current_foot) or word in punctuation_marks:
            valid_indices.append(info['idx'])


    # mask: A tensor of zeros with the same shape as output_probs.
    # the postitions are set corresponding to valid_indices to 1, creating a binary mask where valid word indices have a value of 1.
    mask = torch.zeros_like(output_probs)
    mask[valid_indices] = 1
    # The original output probabilities are element-wise multiplied by the mask.
    adjusted_probs = output_probs * mask

    if last_is_punctuation:
        for punct in punctuation_marks:
            if punct in syllable_dict:
                adjusted_probs[syllable_dict[punct]['idx']] *= 0  # Reduce probability of repeating punctuation

    # Reduce probability of repeating the last word
    if last_word in syllable_dict:
        adjusted_probs[syllable_dict[last_word]['idx']] *= 0.001  # Reduce probability of repeating the last word

    adjusted_probs = torch.where(torch.isnan(adjusted_probs) | (adjusted_probs < 0), torch.tensor(0.0, device=adjusted_probs.device), adjusted_probs)

    if adjusted_probs.sum() > 0:
        adjusted_probs /= adjusted_probs.sum()
    else:
        # If all probabilities have been zeroed out (which shouldn't normally happen, but it's a safeguard), 
        # this creates a uniform distribution where every word has an equal probability.
        adjusted_probs = torch.ones_like(adjusted_probs) / adjusted_probs.numel()

    return adjusted_probs

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")
        
use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

current_foot_idx = 0
current_line = []
words_in_hexameter = 0
last_is_punctuation = False

# NTS: last word does not really do anything here.
last_word = ""

with open(args.outf, 'w') as outf:
    with torch.no_grad():
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = 4
                while word_idx == 4:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word_weights = adjust_prob_distribution(word_weights, current_foot_idx, dactylic_hexameter, syllable_dict, last_word, last_is_punctuation)


with open(args.outf, 'w') as outf:
    with torch.no_grad():
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = 4
                while word_idx == 4:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word_weights = adjust_prob_distribution(word_weights, current_foot_idx, dactylic_hexameter, syllable_dict, last_word, last_is_punctuation)

            word_idx = torch.multinomial(word_weights, 1)[0]

            word = corpus.dictionary.idx2word[word_idx]
            word_syl = syllable_dict[word]['syl']

            if is_transformer_model:
                input = torch.cat([input, word_idx.unsqueeze(0).unsqueeze(0)], 0)
            else:
                input.fill_(word_idx)

            current_line.append(syllable_dict[word]['mac'])
            last_is_punctuation = is_punctuation(word)
            last_word = word  # Update last_word to the current word

            if not last_is_punctuation:
                if fits_foot(word_syl, dactylic_hexameter[current_foot_idx]):
                    current_foot_idx = (current_foot_idx + 1) % len(dactylic_hexameter)
                    words_in_hexameter += 1

                    # If a dactylic hexameter is completed, add '/n'
                    if current_foot_idx == 0:
                        generated_line = ' '.join(current_line).strip()
                        print(f"Debug: Generated line with macrons: {generated_line}")  # Debug print
                        
                        # Remove breves before scanning
                        generated_line_no_breves = remove_breves(generated_line)
                        print(f"Debug: Generated line without breves: {generated_line_no_breves}")  # Debug print
                        
                        scanned_line = scanner.scan(generated_line_no_breves)
                        print(f"Debug: Scanned result: valid={scanned_line.valid}, meter={scanned_line.meter}")  # Debug print
                        
                        if scanned_line.valid and scanned_line.meter == 'hexameter':
                            outf.write(generated_line + '\n')  
                            print(f"Debug: Valid hexameter written: {generated_line}")  # Debug print
                        else:
                            print(f"Debug: Invalid hexameter: {generated_line}")  # Debug print
                        
                        current_line = []
                        words_in_hexameter = 0
                                        

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))