import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
import spacy
import re
from fuzzywuzzy import fuzz
# from nltk.corpus import wordnet as wn
from wordhoard import Synonyms

nlp = spacy.load("en_core_web_sm")

def boxstr_to_boxes(box_str):
    boxes = [[int(y)/1000 for y in x.split(',')] for x in box_str.split(';') if x.replace(',', '').isdigit()]
    print(boxes)
    for x in box_str.split(';'):
        if x.count(',') != 3: # ill-formed bbox
            return None
    return boxes

# need to remove bbox info from output string for parsing to work correctly.
def remove_boxes_from_text(s, indices):
    new_string = []
    current_index = 0
    new_indices = []
    offset = 0
    
    for start, end in indices:
        # Add the part before the current segment to the new string
        new_string.append(s[current_index:start-1]) # assume space before [[a,b,c,d]]
        current_index = end
        offset += (end - start + 1)
        new_indices.append(end - offset)
    
    # Add the remaining part of the string
    new_string.append(s[current_index:])
    
    # Join all parts to form the new string
    new_string = ''.join(new_string)
    
    return new_string, new_indices

# Modified from CogVLM, converts text output string to (object: bbox) pairs
def text_to_dict(text):
    box_matches = list(re.finditer(r'\[\[([^\]]+)\]\]', text))
    box_positions = [match.start() for match in box_matches]

    indices = [[match.start(), match.end()] for match in box_matches]
    clean_text, box_indices = remove_boxes_from_text(text, indices)
    doc = nlp(clean_text)

    noun_phrases = []
    boxes = []

    for match, box_position in zip(box_matches, box_indices):
        nearest_np_start = max([0] + [chunk.start_char for chunk in doc.noun_chunks if chunk.end_char <= box_position])
        noun_phrase = clean_text[nearest_np_start:box_position].strip()
        if "[[" in noun_phrase:
            print("WARNING: nound phrase contains bbox info.\n")
        if noun_phrase and noun_phrase[-1] == '?':
            noun_phrase = clean_text[:box_position].strip()
        
        box_string = match.group(1)
        noun_phrases.append(noun_phrase)
        box = boxstr_to_boxes(box_string)
        if box is not None:
            boxes.append(box)
        else:
            # if one of the bbox is ill-formed, get rid of this try
            return False

    pairs = []
    for noun_phrase, box_string in zip(noun_phrases, boxes):
        pairs.append((noun_phrase.lower(), box_string))
    return dict(pairs)


def find_word_match(text, target_noun, synonym_list=None, threshold=90):
    # Use fuzzy matching to compare the chunk text with the target noun
    # not sure why we need text.replace(" ", "")
    if fuzz.partial_ratio(text.replace(" ", ""), target_noun.replace(" ", "")) >= threshold:
        return True
    # only do synonym matching if fuzzy match fails
    elif synonym_list is not None:
        find_word = [word for word in synonym_list if word == text]
        if len(find_word) != 0: # is synonym
            return True
    return False

# find a noun (object name) that we care about inside a noun phrase
# the target noun should be space separated
# the user should provide synonyms of the target noun to enable synonym matching to speed things up
def find_noun_in_chunk(text, target_noun, synonym_list=None, threshold=90):
    doc = nlp(text)
    matched_chunks = []

    for chunk in doc.noun_chunks:
        nouns_in_chunk = [token.lemma_ for token in chunk if token.pos_ in ['NOUN', 'PROPN']]
        chunk_text = " ".join(nouns_in_chunk)
        if find_word_match(chunk_text, target_noun, synonym_list, threshold):
            matched_chunks.append(chunk.text)
    # if found a match, can return right away
    if len(matched_chunks) > 0:
        return matched_chunks

    for token in doc:
        processed = False
        for chunk in doc.noun_chunks:
            if token in chunk:
                processed = True
        # a special case where a noun token is ignored
        # e.g. "a adler axe" only identifies "a adler" as a noun phrase
        if not processed and token.pos_ in ['NOUN', 'PROPN']:
            if find_word_match(token.text, target_noun, synonym_list, threshold):
                matched_chunks.append(token.text)

    return matched_chunks