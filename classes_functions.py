import hashlib
from random import randint, seed
import math
import itertools
import numpy as np
import pandas as pd
import textdistance as sim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
import statistics
from itertools import product
import swifter
import time
import rapidfuzz

# Compute shinglings given string and size of shinglings
def get_shinglings(lst, k):
    shingle_lst = []

    for string in lst:
        shingles = set()
        for i in range(0, len(string)-k+1 ):
            shingles.add(string[i:i+k])
        
        shingle_lst.append(shingles)
        
    return shingle_lst

# The hash class, used to keep the same randomised hashes throughout the dataset
class hashObject:
    def __init__(self, i, hash_size):
        # The size of the hash
        self.resultSize = hash_size

        # Add (what will be) a random part to get a different hash everytime
        self.randomiser = str(i).zfill(20)[-20:]

    def get_hash(self, element):

        # Compute the raw md5 hash
        self.raw_hash  = hashlib.md5(str(element).encode('utf-8') + self.randomiser.encode('utf-8')).hexdigest()

        # Resize the hash to what we set
        self.stripped_hash = self.raw_hash[-self.resultSize:]

        # Convert to integer for minhash to work
        final_hash = int(self.stripped_hash, 16)

        return final_hash
    
# The minHash class, gets the minHash given a set of shinglings
class minHash:
    def __init__(self, nbr_hashes, hash_size):
        # Number of hashes to use
        self.nbr_hashes = nbr_hashes

        # Init the hash family of functions using a random value for i
        self.hash_functions = [hashObject(randint(0,10000000000), hash_size) for i in range(0, nbr_hashes)]

    def compute_signature_matrix(self, input_sets):

        signature_matrix = []

        for input_set in input_sets:
            # Initialise the signature as an empty array
            set_signature = []

            # Loop through the hash functions as defined above
            for hash_function in self.hash_functions:

                # As minhash looks for the minimum hash, we set the initial value at infinity, with the goal to reduce it as much as we can
                min_hash = math.inf

                # Compute the hash for every element in the set, save it is smaller 
                for element in input_set:
                    h = hash_function.get_hash(element)
                    if h < min_hash:
                        min_hash = h

                # Append the final minimum hash to the list
                set_signature.append(min_hash)

            signature_matrix.append(set(set_signature))

        return signature_matrix

# Function that creates the LSH bands given two signatures and a number of bands
def get_bands_from_signatures(minhash_sig_matrix, nbr_bands): 
    # Next set the number of rows per band
    r = int(len(minhash_sig_matrix[0])/nbr_bands)

    bands = {} 

    for i in range(0, nbr_bands):
        bands[i] = []

    # Populate the bands
    for i in range(0, nbr_bands):
        for signature in minhash_sig_matrix:
            bands[i].append(' '.join(str(x) for x in list(signature)[i*r:r*(i+1)]))
           
    return bands

# Function that creates the buckets for every band and assigns the ids of the corresponding documents
def get_band_buckets(bands, hash_size):

    # Create bucket
    buckets = {}

    # Loop through all the bands
    for band_nbr, band in bands.items():

        randomiser = randint(0,10000000000)

        # Loop through the 2 hashes in the band and categorise them, whether they match or not
        for i in range(len(band)):
            bucket_hash_i = hashObject(randomiser, hash_size).get_hash(band[i])

            if bucket_hash_i in buckets.keys():
                buckets[bucket_hash_i].append(i)
            else:
                buckets[bucket_hash_i] = [i]

    return buckets

def get_candidates_list(buckets):
    potential_candidates = []

    for candidates in buckets.values():
        if len(candidates) > 1:
            potential_candidates.append(list(itertools.combinations(candidates, 2)))

    potential_candidates = set([tuple(sorted(i)) for i in [item for row in potential_candidates for item in row] if (i[0] != i[1])])

    return potential_candidates

def minhash_lsh(documents, k, hash_size, nbr_bands):

    shinglings = get_shinglings(documents, k)

    minHash_instance = minHash(128, hash_size)

    minhash_sig_matrix = minHash_instance.compute_signature_matrix(shinglings)

    bands = get_bands_from_signatures(minhash_sig_matrix, nbr_bands)

    buckets = get_band_buckets(bands, hash_size)

    candidates_list = get_candidates_list(buckets)

    return candidates_list

def get_length(item):
    return len(item)

def get_documents(df):
    documents = df["main_feature"].to_list()
    document_indices = df.index.to_list()
    return documents, document_indices

def initialise_data(df, train_frac):

    df_train = df.sample(frac=train_frac)
    df_test = df.drop(df_train.index)

    shops = df["shop"].to_list()
    brands = df["brand"].to_list()
    model_id = df["modelID"].to_list()
    documents = df["main_feature"].to_list()
    matched_ids = df["matched_id"].to_list()

    documents_train, document_indices_train = get_documents(df_train)
    documents_test, document_indices_test = get_documents(df_test)

    return shops, brands, model_id, documents, matched_ids, documents_train, document_indices_train, documents_test, document_indices_test

def prepapre_df(candidates_list, documents, shops, model_id, brands, matched_ids):

    df = pd.DataFrame(candidates_list, columns=["candidate_1_id", "candidate_2_id"])

    df["candidate_1"] = df["candidate_1_id"].map(documents)
    df["candidate_2"] = df["candidate_2_id"].map(documents)

    df["shop_1"] = df["candidate_1_id"].map(shops)
    df["shop_2"] =  df["candidate_2_id"].map(shops)

    df["brand_1"] = df["candidate_1_id"].map(brands)
    df["brand_2"] = df["candidate_2_id"].map(brands)

    df["id_1"] = df["candidate_1_id"].map(model_id)
    df["id_2"] = df["candidate_2_id"].map(model_id)

    df["id_matched_1"] = df["candidate_1_id"].map(matched_ids)
    df["id_matched_2"] = df["candidate_2_id"].map(matched_ids)

    df["duplicate"] = (df["id_1"] == df["id_2"]).astype("int")
    df["same_shop"] = (df["shop_1"] == df["shop_2"]).astype("int")
    df["same_brand"] = (df["brand_1"] == df["brand_2"]).astype("int")

    # The latter can actually be used for training as it is taken from a regex using the title
    df["same_matched_id"] = (df["id_matched_1"] == df["id_matched_2"]).astype(int)

    # Ready-made features
    df["jaccard"] = df.swifter.progress_bar(False).apply(lambda x: sim.jaccard.normalized_similarity(x['candidate_1'], x['candidate_2']), axis=1)
    df["levensthein"] = df.swifter.progress_bar(False).apply(lambda x: rapidfuzz.distance.Levenshtein.normalized_similarity(x['candidate_1'], x['candidate_2']), axis=1)
    df["cosine"] = df.swifter.progress_bar(False).apply(lambda x: sim.cosine.normalized_similarity(x['candidate_1'], x['candidate_2']), axis=1)
    df["hamming"] = df.swifter.progress_bar(False).apply(lambda x: rapidfuzz.distance.Hamming.normalized_similarity(x['candidate_1'], x['candidate_2']), axis=1)
    df["jarow"] = df.swifter.progress_bar(False).apply(lambda x: rapidfuzz.distance.JaroWinkler.normalized_similarity(x['candidate_1'], x['candidate_2']), axis=1)

    # Manually set same_shop to 0 if duplicate (otherwise the classifier will freak out), same for same_brand (if duplicate, set to 1)
    df.loc[df['duplicate'] == 1, 'same_shop'] = 0
    df.loc[df['duplicate'] == 1, 'same_brand'] = 1

    return df

def classification(X_train, y_train, X_test, y_test, prepped_df_test, y_pred_rejected):

    final_clf = RandomForestClassifier(
        criterion="gini",
        max_depth=100,
        min_samples_split=5, 
        n_estimators=100,
    )

    final_clf.fit(X_train, y_train,)
    raw_pred = final_clf.predict(X_test)

    prepped_df_test["pred"] = raw_pred

    # Manually set pred to 1 if the matched (regex) id's are the same
    prepped_df_test.loc[prepped_df_test['same_matched_id'] == 1, 'pred'] = 1

    predictions = prepped_df_test["pred"].to_list() + y_pred_rejected

    pair_quality = np.sum(np.array(predictions) * np.array(y_test)) / len(X_test)
    pair_completness = np.sum(np.array(predictions) * np.array(y_test)) / np.sum(y_test)

    f1 = f1_score(y_test, predictions)
    f1_star = statistics.harmonic_mean([pair_quality, pair_completness])
    confusion_matrix_arr = confusion_matrix(y_test, predictions)

    return f1, f1_star, confusion_matrix_arr

def main_run(df, train_frac, shingling_size, hash_size, nbr_bands):

    # Initialise Time
    start = time.process_time()

    #Initialise, split and get documents of data
    shops, brands, model_id, documents, matched_ids, documents_train, document_indices_train, documents_test, document_indices_test = initialise_data(df, train_frac)

    # Actually perform LSH on both testing and training datasets
    candidates_list_train_temp = minhash_lsh(documents_train, shingling_size, hash_size, nbr_bands)
    candidates_list_test_temp = minhash_lsh(documents_test, shingling_size, hash_size, nbr_bands)

    # Because indices are reset during LSH, get the real indices back from the original indices
    candidates_list_train = set([tuple((document_indices_train[tup[0]], document_indices_train[tup[1]])) for tup in candidates_list_train_temp])
    candidates_list_test = set([tuple((document_indices_test[tup[0]], document_indices_test[tup[1]])) for tup in candidates_list_test_temp])

    # In order to then create the full confusion matrix (and thus compute F1 scores), one has to grab all pairs of the test dataset that have been rejected by the LSH algorithm 
    # (which hence are automatically marked as non-dupes)
    all_pairs_test = list(product(document_indices_test, repeat = 2))
    all_pairs_without_dupes_test = set([tuple(sorted(i)) for i in all_pairs_test if (i[0] != i[1])])
    pairs_set_non_dupe = list(set(all_pairs_without_dupes_test) - candidates_list_test)

    fraction_of_comparison = len(candidates_list_test)/len(all_pairs_without_dupes_test)

    # Recreate the dataset from pairs rejected by LSH
    rejected_df = pd.DataFrame(pairs_set_non_dupe, columns=["candidate_1_id", "candidate_2_id"])
    rejected_df["id_1"] = rejected_df.swifter.progress_bar(False).apply(lambda x: model_id[x.candidate_1_id], axis=1)
    rejected_df["id_2"] = rejected_df.swifter.progress_bar(False).apply(lambda x: model_id[x.candidate_2_id], axis=1)

    # Get the duplicates from the LSH-rejected pairs
    y_real_rejected = (rejected_df["id_1"] == rejected_df["id_2"]).astype("int").to_list()

    # Manually set their predictions to 0 (non-dupe)
    y_pred_rejected = np.zeros(len(y_real_rejected)).tolist()

    # Create Maps out of the initial lists
    documents_map = {index: value for index, value in enumerate(documents)}
    shops_map = {index: value for index, value in enumerate(shops)}
    model_id_map = {index: value for index, value in enumerate(model_id)}
    brands_map = {index: value for index, value in enumerate(brands)}
    matched_ids_map = {index: value for index, value in enumerate(matched_ids)}

    # Prep the data for Classification (add features, set duplicate marker...)
    prepped_df_train = prepapre_df(list(candidates_list_train), documents_map, shops_map, model_id_map, brands_map, matched_ids_map)
    prepped_df_test = prepapre_df(list(candidates_list_test), documents_map, shops_map, model_id_map, brands_map, matched_ids_map)

    # Create X and y datasets, taking care of adding the  LSH-rejected y'
    features = ["same_shop", "same_brand", "same_matched_id", "jaccard", "levensthein", "cosine", "hamming", "jarow"]
    X_train = prepped_df_train[features]
    y_train = prepped_df_train["duplicate"]
    X_test = prepped_df_test[features]
    y_test = prepped_df_test["duplicate"].to_list() + y_real_rejected

    # Perform classification
    f1, f1_star, confusion_matrix_arr = classification(X_train, y_train, X_test, y_test, prepped_df_test, y_pred_rejected)

    # Get runtime
    runtime_sec = time.process_time() - start

    return f1, f1_star, confusion_matrix_arr, all_pairs_without_dupes_test, runtime_sec, fraction_of_comparison