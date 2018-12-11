from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalMaxPool2D, \
    Dropout, concatenate, Softmax
from keras.models import Model, clone_model
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from time import time


def define_model(height: int, width: int, nclass: int, channels=3) -> Model:
    """
    Defines the architecture for the CNN model
    """
    model_input = Input(shape=(height, width, channels))
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(model_input)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = GlobalMaxPool2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.5)(x)
    model_output = Dense(nclass, activation="softmax")(x)
    model = Model(model_input, model_output)
    return model


def compile_model(model: Model, ngpu):
    """
    Compiles the model
    """
    if ngpu > 1:
        parallel_model = multi_gpu_model(model, gpus=ngpu)
        parallel_model.compile(optimizer=SGD(lr=0.001, nesterov=True),
                               loss="categorical_crossentropy")
        return parallel_model
    else:
        model.compile(optimizer=SGD(lr=0.001, nesterov=True),
                      loss="categorical_crossentropy")
        return model


def refine_model(orig_model: Model, nclass: list, k: int):
    """
    Refines the model for the specialist layers
    """

    # Define the new model
    model = clone_model(orig_model)
    model.set_weights(orig_model.get_weights())

    # Remove the old softmax layer
    for i in range(4):
        model.layers.pop()

    # Set up each of the K branches
    branch_output = []
    for i in range(k):
        x = MaxPooling2D()(model.layers[-1].output)
        x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = GlobalMaxPool2D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(rate=0.5)(x)
        branch_output.append(Dense(nclass[i], activation="relu")(x))

    # Append the K branches into a final softmax layer
    merge = concatenate(branch_output)
    model_output = Softmax()(merge)
    refined_model = Model(model.input, model_output)
    return refined_model


def train_model(model: Model, X_train, y_train, nepoch, batch_size, tol=0.001,
                patience=5):
    early_stopping = EarlyStopping(min_delta=tol, patience=patience)
    orig_classes = np.unique(y_train)
    label_map = dict(zip(range(len(orig_classes)), orig_classes))
    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(y_train.reshape(-1, 1))

    model.fit(
        X_train, y, batch_size, epochs=nepoch, validation_split=0.25,
        callbacks=[early_stopping]
    )
    return label_map


def get_test_pred(model: Model, X_test, batch_size, label_map):
    y_pred = model.predict(X_test, batch_size)
    y_pred = y_pred.argmax(axis=1)
    pred = [label_map[val] for val in y_pred]
    return pred


def create_classification_report(y_test, y_pred):
    """
    Creates a classification report which we can save to disk
    """
    res = precision_recall_fscore_support(y_test, y_pred)
    n = len(res)
    res = [res[i].reshape(-1, 1) for i in range(n)]
    return np.hstack(tuple(res))


def flat_model(X_train, y_train, X_test, y_test, height, width, ngpu,
               nepoch=100, batch_size=128):
    # Train the model
    nclass = len(np.unique(y_train))
    model = define_model(height, width, nclass)
    par_model = compile_model(model, ngpu)
    start_time = time()
    label_map = train_model(par_model, X_train, y_train, nepoch, batch_size)
    train_time = time() - start_time

    # Test the model
    start_time = time()
    y_pred = get_test_pred(par_model, X_test, batch_size, label_map)
    test_time = time() - start_time

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )

    report = [create_classification_report(y_test, y_pred)]

    return {"precision": precision, "recall": recall, "fscore": fscore,
            "train_time": train_time, "test_time": test_time,
            "report": report}


def flatten_list(x):
    return [item for sublist in x for item in sublist]


def map_root_labels(y, label_groups):
    """
    Maps a label vector according to the proscribed partition
    """
    y_root = np.empty_like(y)
    ngroup = len(label_groups)
    for i in range(ngroup):
        idx = np.isin(y, label_groups[i])
        y_root[idx] = i
    return y_root


def update_label_groups(model: Model, X_val, y_val, label_groups, p):
    """
    Updates the label groups based on their performance from the root
    classifier
    """
    # Get the validation predictions from the root
    root_pred = model.predict(X_val).argmax(axis=1)
    y_root = map_root_labels(y_val, label_groups)

    # Compute the conditional probability of each class being mis-labelled
    # so we can add the appropriate classes to the label groups
    ngroup = len(label_groups)
    for j in range(ngroup):
        for i in range(ngroup):
            if i == j:
                continue
            else:
                idx = np.where((root_pred == j) & (y_root == i))[0]
                print(y_val[idx])
                cond_prob = np.bincount(y_val[idx]) / len(idx)
                print(cond_prob)
                new_groups = np.where(cond_prob >= p)
                label_groups[j] = np.append(label_groups[j], new_groups)
                label_groups[j] = label_groups[j].astype(int)
                print(label_groups[j])
    return label_groups


def hierarchical_model(X_train, y_train, X_test, y_test, label_groups,
                       height, width, ngpu, nepoch=100, batch_size=128):

    # Re-map the initial set of labels for the first layer
    ngroup = len(label_groups)
    y_root = map_root_labels(y_train, label_groups)

    # Train the root model
    root_model = define_model(height, width, ngroup)
    root_par_model = compile_model(root_model, ngpu)
    start_time = time()
    root_map = train_model(root_par_model, X_train, y_root, nepoch, batch_size)

    # # Update the label groups as necessary based on the performance
    # # of the root classifier
    # label_groups = update_label_groups(root_model, X_val, y_val, label_groups,
    #                                    p)

    # # Train a model for each of the leaves
    nclass = [len(label_groups[i]) for i in range(ngroup)]
    # leaf_model = refine_model(root_par_model, nclass, ngroup)
    # leaf_par_model = compile_model(leaf_model)
    # # label_maps = [dict()] * ngroup
    # train_model(leaf_par_model, X_train, y_train, nepoch, batch_size)

    nleaf = len([nclass[i] for i in range(ngroup) if nclass[i] > 1])
    leaf_idx = [i for i in range(ngroup) if nclass[i] > 1]
    label_maps = [dict()] * nleaf
    leaf_models = [define_model(height, width, nclass[i]) for i in
                   range(ngroup) if nclass[i] > 1]
    for (i, val) in enumerate(leaf_idx):
        idx = np.isin(y_train, label_groups[val])
        leaf_models[i] = compile_model(leaf_models[i], ngpu)
        label_maps[i] = train_model(leaf_models[i], X_train[idx], y_train[idx],
                                    nepoch, batch_size)
    train_time = time() - start_time

    # Get the root predictions
    start_time = time()
    root_pred = get_test_pred(root_par_model, X_test, batch_size, root_map)
    root_pred = np.array(root_pred)

    # Using the root predictions, make the final predictions
    y_pred = np.empty_like(y_test)
    idx = [np.where(root_pred == i)[0] for i in range(ngroup)]
    lone_leaf = np.setdiff1d(np.arange(ngroup), leaf_idx)

    # Get the predictions for the lone classes
    root_map = dict(zip(range(ngroup), label_groups))
    for label in lone_leaf:
        y_pred[idx[label]] = np.repeat(root_map[label], len(idx[label]))

    # Get the predictions for the meta-classes
    for (i, val) in enumerate(leaf_idx):
        if len(idx[val]) == 0:
            continue
        y_pred[idx[val]] = get_test_pred(leaf_models[i], X_test[idx[val]],
                                         batch_size, label_maps[i])
    # labels = flatten_list(label_groups)
    # label_map = dict(zip(range(len(nclass)), labels))
    # y_pred = get_test_pred(leaf_par_model, X_test, batch_size, label_map)
    test_time = time() - start_time

    # reports = [create_classification_report(y_test[idx[i]], y_pred[idx[i]])
    #            for i in range(ngroup)]

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    return {"precision": precision, "recall": recall, "fscore": fscore,
            "train_time": train_time, "test_time": test_time}
