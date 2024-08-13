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

parser = argparse.ArgumentParser(description='PyTorch Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/corpus_10mio_10k', # CHANGE
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/model_10mio_10k.pt', # CHANGE
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='./generated/ten_million/generated_10mio_10k_elision.txt', # CHANGE twice
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111, #default 1111
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--temperature', type=float, default=1.0, # CHANGE
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
ending_punct_marks = {'.', '!', '?'}
vowels = 'aeiouyăĕĭŏŭy̆āēīōūȳ'

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

def is_ending_punct(word):
    return word in ending_punct_marks

"""
If a word ends in a vowel (e.g. vento), vowels = 'aeiouyăĕĭŏŭy̆āēīōūȳ', or a vowel + m (e.g. vitam), AND the next word begins with a vowel or h, the vowel, and the m/h are not scanned and generally not pronounced. So then we leave the word as it is but attach a "*" right after the elided syllable. Note that the meter count is not adapted correctly though from what I see.
"""

def check_elision(word1, word2, syllable_dict):
    if word1 not in syllable_dict or word2 not in syllable_dict:
        return False
    
    word1_mac = syllable_dict[word1]['mac']
    word2_mac = syllable_dict[word2]['mac']
    
    # Check if word1 ends with a vowel or vowel+m
    if word1_mac[-1] in vowels or (len(word1_mac) > 1 and word1_mac[-2] in vowels and word1_mac[-1] == 'm'):
        # Check if word2 starts with a vowel or h
        if word2_mac[0] in vowels or word2_mac[0] == 'h':
            # If the last syllable of word1 is long by nature and ends with 'm', don't elide
            if syllable_dict[word1]['syl'][-1] == 'l' and word1_mac[-1] == 'm':
                return False
            return True
    return False


def apply_elision(word, syllable_dict):
    word_mac = syllable_dict[word]['mac']
    word_syl = syllable_dict[word]['syl']
    
    if word_mac[-1] in vowels:
        return word_mac + '*', word_syl[:-1]
    elif len(word_mac) > 1 and word_mac[-2] in vowels and word_mac[-1] == 'm':
        return word_mac[:-1] + '*', word_syl[:-1]
    else:
        return word_mac, word_syl

# For some reason, there sometimes occur dobuled initial letters for words with an asterisk. Fix around for now:
def process_text(text):
    def modify_word(word):
        if '*' in word:
            # Remove the asterisk for processing
            # In case * should be removed: word = word.replace('*', '')
            # Check if the first letter and second letter are the same
            if len(word) > 1 and word[0].lower() == word[1].lower():
                # Remove the second letter
                word = word[0] + word[2:]
        return word

    # Split the text into lines
    lines = text.splitlines()
    modified_lines = []

    for line in lines:
        # Split the line into words
        words = re.findall(r'\S+', line)
        # Process each word
        modified_words = [modify_word(word) for word in words]
        # Rebuild the line with preserved spacing
        modified_line = re.sub(r'\S+', lambda match: modify_word(match.group(0)), line)
        modified_lines.append(modified_line)
    
    # Join modified lines with line breaks
    return '\n'.join(modified_lines)
    

def adjust_prob_distribution(output_probs, current_foot_idx, dactylic_hexameter, syllable_dict, last_word, last_is_punctuation, last_is_ending_punct):
    current_foot = dactylic_hexameter[current_foot_idx]
    valid_indices = []

    for word, info in syllable_dict.items():
        word_syl = info['syl']
        if check_elision(last_word, word, syllable_dict):
            _, word_syl = apply_elision(last_word, syllable_dict)
        
        # Same as before
        if fits_foot(word_syl, current_foot) or word in punctuation_marks:
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
words_in_hexameter = 0
last_is_punctuation = False
last_is_ending_punct = False
last_word = ""
lines = []

with open(args.outf, 'w', encoding='utf-8') as outf:
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

            word_weights = adjust_prob_distribution(word_weights, current_foot_idx, dactylic_hexameter, syllable_dict, last_word, last_is_punctuation, last_is_ending_punct)

            word_idx = torch.multinomial(word_weights, 1)[0]

            word = corpus.dictionary.idx2word[word_idx]
            word_syl = syllable_dict[word]['syl']

            # Check for elision
            if check_elision(last_word, word, syllable_dict):
                last_word_mac, _ = apply_elision(last_word, syllable_dict)
                outf.seek(outf.tell() - len(last_word))
                outf.write(last_word_mac)

            if is_transformer_model:
                input = torch.cat([input, word_idx.unsqueeze(0).unsqueeze(0)], 0)
            else:
                input.fill_(word_idx)

            # Capitalizing the first word or the word after punctuation
            if i == 0 or last_is_ending_punct:
                word_mac = syllable_dict[word]['mac'].capitalize()
            else:
                word_mac = syllable_dict[word]['mac']

            # Fixing the punctuation attachment
            if is_punctuation(word) or i == 0:
                outf.write(word_mac)
            else:
                outf.write(' ' + word_mac)

            last_is_punctuation = is_punctuation(word)
            last_is_ending_punct = is_ending_punct(word)
            last_word = word  # Update last_word to the current word

            if not last_is_punctuation:
                if fits_foot(word_syl, dactylic_hexameter[current_foot_idx]):
                    current_foot_idx = (current_foot_idx + 1) % len(dactylic_hexameter)
                    words_in_hexameter += 1
                    
                    # If a dactylic hexameter is completed, add a newline
                    if current_foot_idx == 0:
                        outf.write('\n')
                        words_in_hexameter = 0

                    # Ensure flushing after each word to make sure it's written
                    # outf.flush()

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

            lines.append(word_mac)
            

def convert_to_utf8_handling_errors(input_file_path, output_file_path):
    try:
        # Open the input file with UTF-8 encoding, ignoring errors
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as infile:
            content = infile.read()
            # Write the content to the output file with UTF-8 encoding
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(content)
        print(f"File successfully converted to UTF-8 and saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to convert file: {e}")

#input_file_path = '/home/pricie/cclstudent10/pytorch_model/ophelia/examples/word_language_model/generated/ten_million/generated_10mio_10k_elision_2.txt' # CHANGE
#output_file_path = '/home/pricie/cclstudent10/pytorch_model/ophelia/examples/word_language_model/generated/ten_million/generated_10mio_10k_elision_2_utf8.txt' # CHANGE

# Convert the generated text to UTF-8
input_file_path = args.outf
output_file_path = input_file_path.replace('.txt', '_utf8.txt')
convert_to_utf8_handling_errors(input_file_path, output_file_path)

# Process the generated text
with open(output_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

processed_text = process_text(text)

# Save the processed text
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(processed_text)
