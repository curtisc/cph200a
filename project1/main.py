
import argparse
import json
from csv import DictReader
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import matplotlib.pyplot as plt

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plco_data_path",
        default="plco/lung_prsn.csv",
        help="Location of PLCO csv",
    )

    parser.add_argument(
        "--learning_rate",
        default=1,
        type=float,
        help="Learning rate to use for SGD",
    )

    parser.add_argument(
        "--regularization_lambda",
        default=0,
        type=float,
        help="Weight to use for L2 regularization",
    )

    parser.add_argument(
        "--batch_size",
        default=4000,
        type=int,
        help="Batch_size to use for SGD"
    )

    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="number of epochs to use for training"
    )

    parser.add_argument(
        "--results_path",
        default="results.json",
        help="Where to save results"
    )

    parser.add_argument(
        "--verbose",
        default=False,
        help="Whether to print verbose output"
    )

    return parser

def load_data(args: argparse.Namespace) -> ([list, list, list]):
    '''
    Load PLCO data from csv file and split into train validation and testing sets.
    '''
    reader = DictReader(open(args.plco_data_path,"r"))
    rows = [r for r in reader]
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)  # Keep data split consistent for fair comparison
    random.shuffle(rows)
    train, val, test = rows[:NUM_TRAIN], rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL], rows[NUM_TRAIN+NUM_VAL:]

    return train, val, test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    print("Loading data from {}".format(args.plco_data_path))
    train, val, test = load_data(args)

    # Define someway to defined what features your model should use
    # lung_cancer is the label
    feature_config = {
        "numerical": ["age"],
        "categorical": [#"sex", 
                        #"race7", 
                        #"hispanic_f", 
                        "educat", 
                        "marital", 
                        "occupat", 
                        "cig_stat", 
                        "pack_years",
                        "cig_years",
                        "cigpd_f", 
                        "fh_cancer", 
                        "lung_fh", 
                        "cigar"
                        ]
    }

    print("Initializing vectorizer and extracting features")
    # Implement a vectorizer to convert the questionare features into a feature vector
    plco_vectorizer = Vectorizer(feature_config)

    # Fit the vectorizer on the training data (i.e. compute means for normalization, etc)
    plco_vectorizer.fit(train)

    # Featurize the training, validation and testing data
    train_X = plco_vectorizer.transform(train)
    val_X = plco_vectorizer.transform(val)
    test_X = plco_vectorizer.transform(test)

    train_Y = np.array([int(r["lung_cancer"]) for r in train])
    val_Y = np.array([int(r["lung_cancer"]) for r in val])
    test_Y = np.array([int(r["lung_cancer"]) for r in test])

    print("Training model")

    # Initialize and train a logistic regression model
    model = LogisticRegression(num_epochs=args.num_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, regularization_lambda=args.regularization_lambda, verbose=True)
    print(args)
    model.fit(train_X, train_Y, val_X, val_Y)

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(model.train_losses) + 1), model.train_losses, label='Training Loss', marker='o', markersize=3)
    plt.plot(range(1, len(model.val_losses) + 1), model.val_losses, label='Validation Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    print("Loss curves saved to loss_curves.png")

    print("Evaluating model")

    pred_train_Y = model.predict_proba(train_X)
    pred_val_Y = model.predict_proba(val_X)

    results = {
        "train_auc": roc_auc_score(train_Y, pred_train_Y),
        "val_auc": roc_auc_score(val_Y, pred_val_Y)
    }

    print(results)

    print("Saving results to {}".format(args.results_path))

    json.dump(results, open(args.results_path, "w"), indent=True, sort_keys=True)

    print("Done")

    return results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)