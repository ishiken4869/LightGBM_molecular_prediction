
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
TARGET_COL = "HOMO/LUMO gap"
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


def model_builder(**model_params):
    estimator = LGBMRegressor()
    estimator.set_params(**model_params)
    return dc.models.SklearnModel(estimator)


def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, required=True, help="trainig data file(csv)")
    parser.add_argument("-target_col", type=str, required=True)
    parser.add_argument("-smiles_col", type=str, default="smiles")
    args = parser.parse_args()
    '''

    featurizer = dc.feat.RDKitDescriptors()

    # 学習データの読み込み
    loader = dc.data.CSVLoader(tasks=[TARGET_COL],
                               feature_field=SMILES_COL,
                               featurizer=featurizer)

    if DATA_EXIST == True:
        dataset = dc.data.DiskDataset(DATA_DIR)
    else:
        dataset = loader.create_dataset(TRAIN, data_dir=DATA_DIR)

    print("Data loaded.")

    splitter = dc.splits.IndexSplitter()

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2)
    transformers = [dc.trans.NormalizationTransformer(transform_y=True,dataset=train_dataset)]

    for transformer in transformers:
        train_dataset = transformer.transform(train_dataset)
        valid_dataset = transformer.transform(valid_dataset)

    print(train_dataset.X.shape)
    print(valid_dataset.X.shape)
    print(test_dataset.X.shape)

    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    optimizer = dc.hyper.GridHyperparamOpt(model_builder)

    params_dict = {
            #'boosting_type': ['rf'],
            'num_leaves': [80],
            'criterion': ['absolute_error'],
            'max_depth': [100],
            'min_child_samples': [60],
            'learning_rate': [0.01],
            'n_estimators': [32000],
            'reg_lambda': [0],
            'colsample_bytree': [0.4],
            'n_jobs': [multiprocessing.cpu_count()],
            }

    # ハイパーパラメータを決定する際は、nb_epochを変更する
    # ハイパーパラメータが決定している場合は、nb_epoch=1で良い
    best_model, best_model_hyperparams, all_model_results = optimizer.hyperparam_search(params_dict=params_dict, 
                                                                                        train_dataset=train_dataset, 
                                                                                        valid_dataset=valid_dataset, 
                                                                                        output_transformers=transformers, 
                                                                                        metric=metric, 
                                                                                        use_max=False, 
                                                                                        )

    # モデルの保存
    # best_model.save()
    save_to_disk(best_model, MODEL_DIR + "/model.joblib")
    print("model saved.")

    #loaded_model = dc.models.SklearnModel(model=LGBMRegressor(), model_dir=MODEL_DIR)
    loaded_model = load_from_disk(MODEL_DIR + "/model.joblib")
    print("model loaded.")

    time_sta = time.perf_counter()
    predicted_test = loaded_model.predict(test_dataset, transformers=transformers)
    time_end = time.perf_counter()


    tim = time_end - time_sta
    print(tim)

    true_test = test_dataset.y


    print("MAE: {}".format(dc.metrics.mean_absolute_error(true_test, predicted_test)))

    print("best hyperparamers: {}".format(best_model_hyperparams))

    plt.scatter(true_test, predicted_test, s=1)
    plt.xlabel('label')
    plt.ylabel('output')
    plt.title(TARGET_COL)
    plt.grid(True)
    plt.savefig("../scatter/" + PROPERTY + "/scatter.png")

    print("scatter plot saved.")

    #from joblib import dump, load
    #dump(best_model, "../model/" + PROPERTY + "/model.joblib")



if __name__ == "__main__":
    main()