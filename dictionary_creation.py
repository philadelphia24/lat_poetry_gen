import json
import torch
from cltk.prosody.lat.macronizer import Macronizer
import re
import data  # Import the Dictionary and Corpus classes from data.py

# Initialize the Macronizer instance
macronizer = Macronizer(tagger="tag_ngram_123_backoff")


def get_syllable_lengths(word):
    syllable_lengths = []
    macronized_word = macronize_word(word)
    diphthongs = ['aē', 'aū', 'eī', 'eū', 'oē', 'uī', 'aī', 'oī', 'oū']
    vowels = 'aeiouyāēīōūȳ'
    consonants = 'bcdfgjklmnpqrstvwxz' # Removed 'h'
    short_vowel_map = {'a': 'ă', 'e': 'ĕ', 'i': 'ĭ', 'o': 'ŏ', 'u': 'ŭ', 'y': 'y̆'}

    # Consonant combinations that don't shorten a preceding vowel
    non_shortening_combinations = ['bl', 'br', 'cl', 'chl', 'cr', 'chr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tr', 'thr']
    
    i = 0
    while i < len(macronized_word):
        if i < len(macronized_word) - 1:
            diphthong = macronized_word[i:i+2]
            if diphthong == 'qu':
                i += 2
            elif diphthong in diphthongs:
                syllable_lengths.append('l')
                i += 2
            else:
                char = macronized_word[i]
                next_char = macronized_word[i+1]
                if char in 'āēīōūȳ':
                    syllable_lengths.append('l')
                elif char in 'aeiouy':
                    # **Added this block for handling vowels at the end of the word**
                    if i == len(macronized_word) - 2 and next_char in 'aeiouy':
                        syllable_lengths.append('s')
                        macronized_word = macronized_word[:i] + short_vowel_map.get(char, char) + macronized_word[i + 1:]
                    # New rule: vowel followed by two or more consonants, or x or z
                    # Check if there are at least two more characters after the current one
                    elif i < len(macronized_word) - 2:
                        next_two_chars = macronized_word[i+1:i+3]
                        next_three_chars = macronized_word[i+1:i+4]
                        # Note: next_two_chars[1] is the second character of next_two_chars (i.e. the second consonant)
                        if ((next_char in consonants and next_two_chars[1] in consonants and 
                             next_two_chars not in non_shortening_combinations and 
                             next_three_chars[:3] not in non_shortening_combinations) or 
                            next_char in 'xz'):
                            
                            syllable_lengths.append('l')
                            macronized_word = macronized_word[:i] + macronized_word[i].translate(str.maketrans('aeiouy', 'āēīōūȳ')) + macronized_word[i+1:]

                        # A vowel followed by another vowel (not forming a diphthong or being followed by an "h") is short.
                        elif next_char in vowels and diphthong not in diphthongs:
                            syllable_lengths.append('s') # e.g. bĕatus
                            macronized_word = macronized_word[:i] + short_vowel_map.get(char, char) + macronized_word[i+1:]
                        elif next_char == 'h':
                            syllable_lengths.append('s') # Default to unk if unsure
                            macronized_word = macronized_word[:i] + short_vowel_map.get(char, char) + macronized_word[i+1:]
                        else:
                            syllable_lengths.append('u')
                    else:
                        syllable_lengths.append('u') # Default to unk if at end of word
                i += 1

        else:
            char = macronized_word[i]
            if char in 'āēīōūȳ':
                syllable_lengths.append('l')
            elif char in 'aeiouy':
                syllable_lengths.append('u')
            i += 1
    return ''.join(syllable_lengths), macronized_word


def macronize_word(word):
    macronized_word = macronizer.macronize_text(word)
    replacements = {
        'ae': 'aē',
        'au': 'aū',
        'ei': 'eī',
        'eu': 'eū',
        'oe': 'oē',
        'ui': 'uī',
        'ai': 'aī', # early Latin
        'oi': 'oī', # early Latin
        'ou': 'oū'  # early Latin
    }
    for old, new in replacements.items():
        macronized_word = macronized_word.replace(old, new)
    return macronized_word


def create_syllable_dict(dictionary, output_file, manual_updates):
    syllable_dict = {}
    # List of tokens to exclude from macronization
    excluded_tokens = {"<sos>", "<eos>", "<unk>", ",", ".", "!", "?", ";", ":"}

    for word, index in dictionary.word2idx.items():
        if word not in excluded_tokens:
            syllable_lengths, macronized_word = get_syllable_lengths(word)
            syllable_dict[word] = {
                'idx': index,
                'mac': macronized_word,
                'syl': syllable_lengths
            }
        else:
            syllable_dict[word] = {
                'idx': index,
                'mac': word,
                'syl': ''
            }
            
    # MEW: Apply manual updates
    syllable_dict = apply_manual_updates(syllable_dict, manual_updates)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(syllable_dict, f, ensure_ascii=False, indent=4)
    print(f'Syllable dictionary saved to {output_file}')

# NEW
def apply_manual_updates(syllable_dict, manual_updates):
    for word, updates in manual_updates.items():
        if word in syllable_dict:
            syllable_dict[word].update(updates)
    return syllable_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Syllable Dictionary from the Dictionary in data.py')
    parser.add_argument('--output_file', type=str, default='./syllable_dictionaries/syllable_dict_keras.json',  # CHANGE 
                        help='Output file path for the syllable dictionary')
    parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
    args = parser.parse_args()
    
    # Load the Corpus and get the dictionary (assumes data path is hardcoded or specified elsewhere)
    corpus = data.Corpus('./data/lat_poetry_full_vocabulary_normal')  # CHANGE
    dictionary = corpus.dictionary

    # NEW: Define your manual updates: āēīōūȳ, ăĕĭŏŭy̆
    manual_updates = {
        "iuste":  {"mac": "jūste", "syl": "lu"},
        "subiungit": {"mac": "sūbjūngit", "syl": "llu"},
        "iuramentum": {"mac": "juramēntum", "syl": "uulu"},
        "iudicis": {"mac": "judicis", "syl": "uuu"},
        "iusti": {"mac": "jūstī", "syl": "ll"},
        "huiusmodi": {"mac": "hūjūsmodī", "syl": "llul"},
        "iustitiae": {"mac": "jūstitĭaē", "syl": "lusl"},
        "cuiuslibet": {"mac": "cūjūslibet", "syl": "lluu"},
        "adiutorium": {"mac": "ādjūtōrĭum", "syl": "lllsu"},
        "iudicem": {"mac": "jūdicem", "syl": "luu"},
        "uniuscuiusque": {"mac": "ūnīūscūjūsque", "syl": "lllllu"},
        "iusto": {"mac": "jūsto", "syl": "lu"},
        "iustorum": {"mac": "jūstōrum", "syl": "llu"},
        "iugo": {"mac": "jugō", "syl": "ul"},
        "iugiter": {"mac": "jūgiter", "syl": "luu"},
        "cuiusdam": {"mac": "cūjūsdam", "syl": "llu"},
        "iudex": {"mac": "jūdēx", "syl": "ll"},
        "coniunctio": {"mac": "cōnjūnctĭo", "syl": "llsu"},
        "cuiuscumque": {"mac": "cūjūscūmque", "syl": "lllu"},
        "iudices": {"mac": "jūdicēs", "syl": "lul"},
        "coniugio": {"mac": "conjugĭō", "syl": "uusl"},
        "iniuste": {"mac": "īnjūstē", "syl": "lll"},
        "iugum": {"mac": "jugum", "syl": "uu"},
        "coniuncta": {"mac": "cōnjūnctā", "syl": "lll"},
        "coniuncti": {"mac": "cōnjūnctī", "syl": "lll"},
        "huiuscemodi": {"mac": "hūjūscemodī", "syl": "lluul"},
        "coniunctione": {"mac": "cōnjūnctĭōne", "syl": "llslu"},
        "iudae": {"mac": "jūdaē", "syl": "ll"},
        "iudaei": {"mac": "jūdaēī", "syl": "lll"},
        "iudaeorum": {"mac": "jūdaēōrum", "syl": "lllu"},
        "iudaeis": {"mac": "jūdaēīs", "syl": "lll"},
        "iuris": {"mac": "jūris", "syl": "lu"},
        "coniungitur": {"mac": "cōnjūngitur", "syl": "lluu"},
        "diuersa": {"mac": "dīvērsa", "syl": "llu"},
        "iudicare": {"mac": "jūdicāre", "syl": "lulu"},
        "iudas": {"mac": "jūdās", "syl": "ll"},
        "iniustitia": {"mac": "injūstitĭa", "syl": "ulusu"},
        "diuina": {"mac": "dīvīna", "syl": "llu"},
        "diuersis": {"mac": "dīvērsīs", "syl": "lll"},
        "siue": {"mac": "sīve", "syl": "lu"},
        "diuisio": {"mac": "dīvīsĭō", "syl": "llsl"},
        "iustificationem": {"mac": "jūstificātĭōnem", "syl": "luulslu"},
        "diuisione": {"mac": "dīvīsĭōne", "syl": "llslu"},
        "diuisionem": {"mac": "dīvīsĭōnem", "syl": "llslu"},
        "relatiua": {"mac": "relātīva", "syl": "ullu"},
        "uniuersalis": {"mac": "ūnivērsālis", "syl": "lullu"},
        "uniuversale": {"mac": "ūnivērsalē", "syl": "lulul"},
        "iustificatio": {"mac": "jūstificātĭō", "syl": "luulsl"}, #20k covered until here.
        "diuinitas": {"mac": "dīvīnitās", "syl": "llul"},
        "oliueti": {"mac": "olīvētī", "syl": "ulll"},
        "iuunem": {"mac": "juvenem", "syl": "uuu"},
        "coniugationes": {"mac": "cōnjugatĭones", "syl": "luusuu"},
        "iuit": {"mac": "īvit", "syl": "lu"},
        "viuat": {"mac": "vīvat", "syl": "lu"},
        "ioannes": {"mac": "", "syl": ""},
        "<UNK>": {"mac": "<unk>", "syl": ""},
        "<EOS>": {"mac": "<eos>", "syl": ""},
       # "": {"mac": "", "syl": ""},
        # Add more manual updates here
    }
    ###############################
    # OPTIONAL TO CHECK WHAT THE DICTIONARY CLASS DOES
    
    # Print word2idx
    # print("word2idx:")
    # for word, idx in dictionary.word2idx.items():
    #   print(f"{word}: {idx}")
    
    # Print idx2word
    # print("\nidx2word:")
    # for idx, word in enumerate(dictionary.idx2word):
    #    print(f"{idx}: {word}")

    # Save word2idx to a JSON file
    # with open('word2idx.json', 'w', encoding='utf-8') as f:
     #   json.dump(dictionary.word2idx, f, ensure_ascii=False, indent=4)
    
    # Save idx2word to a JSON file
    # with open('idx2word.json', 'w', encoding='utf-8') as f:
    #    json.dump(dictionary.idx2word, f, ensure_ascii=False, indent=4)
    
    ###############################
    
    # Create the syllable dictionary
    create_syllable_dict(dictionary, args.output_file, manual_updates) #NEW: ", manual_updates"