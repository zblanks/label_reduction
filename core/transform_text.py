import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from core.contractions import CONTRACTION_MAP
import unicodedata


def _strip_html_tags(text: str) -> str:
    """
    Strips HTML tags
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def _remove_accented_chars(text: str) -> str:
    """
    Removes characters with accents
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    text = text.decode('utf-8', 'ignore')
    return text


def _expand_contractions(text: str, contraction_mapping: dict):
    """
    Expands contractions (ex: isn't -> is not)
    """
    contractions_pattern = re.compile(
        '({})'.format('|'.join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def _remove_special_characters(text: str, remove_digits: bool):
    """
    Removes special characters
    """
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)

    # Account for the case of []'s in the data
    text = re.sub(r'[\[\]]', '', text)
    return text


# def _simple_stemmer(text: str):
#     """
#     Stems the words
#     """
#     ps = nltk.porter.PorterStemmer()
#     text = ' '.join([ps.stem(word) for word in text.split()])
#     return text
#
#
# def _lemmatize_text(text: str):
#     """
#     Lemmatizes the text
#     """
#     nlp = spacy.load('en')
#     text = nlp(text)
#     text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text
#                      for word in text])
#     return text


def _remove_stopwords(text: str, is_lower_case: bool):
    """
    Remove common stop-words
    """
    # Get the NLTK tokenizer and stop-words
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip().lower() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token
                           not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower()
                           not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def process_text(text: str, remove_digits=True, is_lower_case=True,
                 contraction_mapping=CONTRACTION_MAP):
    """
    Performs all of the pre-processing steps for the text
    """

    # Strip the HTML tags
    text = _strip_html_tags(text)

    # Remove accented text
    text = _remove_accented_chars(text)

    # Remove contractions
    text = _expand_contractions(text, contraction_mapping)

    # Remove special characters
    special_char_pattern = re.compile(r'([{.(-)!}])')
    text = special_char_pattern.sub(" \\1 ", text)
    text = _remove_special_characters(text, remove_digits)

    # # Stem and lemmatize the text
    # text = _simple_stemmer(text)
    # text = _lemmatize_text(text)

    # Remove the stop-words
    text = _remove_stopwords(text, is_lower_case)
    return text
