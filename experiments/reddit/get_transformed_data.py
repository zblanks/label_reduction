from core.transform_text import process_text
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import argparse
from os import path
import spacy
import h5py
from tqdm import tqdm


def get_labels(df: pd.DataFrame):
    """
    Gets the labels provided in the data
    """

    # Get the unique labels in the data
    uniq_labels = df.subreddit.unique()

    # Define a label map
    label_map = dict(zip(uniq_labels, range(len(uniq_labels))))

    # Go through the vector of labels and map them to their integer
    # values
    labels = df.subreddit.values
    return np.array([label_map[val] for val in labels])


def embed_words(text: str, nlp_vec) -> np.ndarray:
    """
    Get the embedding vectors for each of the words and create
    a Continuous Bag of Words (CBOW) mapping
    """

    # Define a placeholder for the vector embeddings
    word_vecs = []

    # Go through each word in the text and get the embedding vector
    tokens = nlp_vec(text)
    for token in tokens:
        word_vecs.append(token.vector.reshape(1, -1))

    # Get the final matrix of vector and get a CBOW mapping
    word_mat = np.concatenate(word_vecs, axis=0)
    return word_mat.mean(axis=0).reshape(1, -1)


def get_text_vectors(df: pd.DataFrame):
    """
    Gets the text vectors for each sample
    """
    # First combine the title and the body-post
    df['text'] = df['title'] + ' ' + df['selftext']
    text_vec = df['text'].values

    # Define the spaCy processor once so that we don't have to keep calling
    # it
    nlp_vec = spacy.load('en_vectors_web_lg')

    # Process all of the text in parallel to speed things up
    with Parallel(n_jobs=-1, verbose=5) as p:
        processed_text = p(delayed(process_text)(text) for text in text_vec)

    # Since the spaCy value cannot be pickled, we will have to get the
    # CBOW embeddings serially
    doc_vecs = [embed_words(text, nlp_vec) for text in tqdm(processed_text)]

    # Convert the document vectors into a single matrix
    return np.concatenate(doc_vecs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/reddit")
    args = vars(parser.parse_args())

    # Get the Reddit DataFrame
    df = pd.read_csv(path.join(args['wd'], 'rspct.tsv'), sep='\t')

    # Get the labels
    y = get_labels(df)

    # Get the main data
    X = get_text_vectors(df)

    # Save the results to disk
    f = h5py.File(path.join(args['wd'], 'data.h5'), 'w')
    f.create_dataset('X', data=X)
    f.create_dataset('y', data=y)
    f.close()


if __name__ == '__main__':
    main()
