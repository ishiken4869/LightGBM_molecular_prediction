#import argparse
import os
import deepchem as dc
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV
import multiprocessing
import time
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TRAIN = "../data/smiles_all_property.csv"
TARGET_COL = "wavelength"
SMILES_COL = "smiles"
PROPERTY = TARGET_COL


if TARGET_COL == "Decomposition Energy":
    PROPERTY = "Decomposition_Energy"
elif TARGET_COL == "HOMO/LUMO gap":
    PROPERTY = "HOMOLUMO_gap"

MODEL_DIR = "../model/" + PROPERTY
LOG_DIR = "../log/" + PROPERTY

DATA_DIR = "../data/" + PROPERTY

DATA_EXIST = os.path.exists(DATA_DIR + "/tasks.json")

def main():
    featurizer = dc.feat.RDKitDescriptors()

    # 学習データの読み込み
    loader = dc.data.CSVLoader(tasks=[TARGET_COL],
                               feature_field=SMILES_COL,
                               featurizer=featurizer)

    if DATA_EXIST == True:
        dataset = dc.data.DiskDataset(DATA_DIR)
    else:
        dataset = loader.create_dataset(TRAIN, data_dir=DATA_DIR)

    splitter = dc.splits.IndexSplitter()

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2)

    transformers = [dc.trans.NormalizationTransformer(transform_y=True,dataset=train_dataset)]

    print(test_dataset.X.shape)

    loaded_lgbm_model = load_from_disk(MODEL_DIR + "/model.joblib")

    predicted_test = loaded_lgbm_model.predict(test_dataset, transformers=transformers)
    true_test = test_dataset.y

    plt.scatter(true_test, predicted_test, s=1)
    plt.xlabel('label')
    plt.ylabel('output')
    plt.title(TARGET_COL)
    plt.grid(True)
    plt.savefig("../scatter/" + PROPERTY + "/scatter.png")


if __name__ == "__main__":
    main()