import os
import sys
import glob
import time
import argparse
import csv
import logging
import pprint
import pyshark
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from lucid_dataset_parser import (
    parse_labels,
    process_live_traffic,
    dataset_to_list_of_fragments
)
from util_functions import (
    static_min_max,
    normalize_and_padding,
    TIME_WINDOW
)

OUTPUT_FOLDER = "./output/"
VAL_HEADER     = ['Model','Samples','Accuracy','F1Score','Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model','Time','Packets','Samples','DDOS%','Accuracy','F1Score','TPR','FPR','TNR','FNR','Source']

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_hdf5_dataset(path: str, packet_feature_index: int = 0,
                      x_key: str = "set_x", y_key: str = "set_y"):
    """Load flows and labels; extract only one feature dimension."""
    with h5py.File(path, "r") as f:
        raw = f[x_key][:]
        y   = f[y_key][:]
    X = raw[:, :, packet_feature_index]
    return X, y


def build_logistic_model(input_dim: int, learning_rate: float = 0.001) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(1, activation="sigmoid", name="output"),
    ], name="logistic")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def report_results(Y_true, Y_pred, packets, model_name, data_source, writer):
    ddos_rate = sum(Y_pred) / Y_pred.shape[0]
    accuracy  = accuracy_score(Y_true, Y_pred)
    f1        = f1_score(Y_true, Y_pred)

    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0,1]).ravel()

    # avoid zero‐division
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    row = {
      'Model':    model_name,
      'Time':     f"{packets:.3f}",
      'Packets':  packets,
      'Samples':  Y_pred.shape[0],
      'DDOS%':    f"{ddos_rate:0.3f}",
      'Accuracy': f"{accuracy:0.4f}",
      'F1Score':  f"{f1:0.4f}",
      'TPR':      f"{tpr:0.4f}",
      'FPR':      f"{fpr:0.4f}",
      'TNR':      f"{tnr:0.4f}",
      'FNR':      f"{fnr:0.4f}",
      'Source':   data_source
    }
    writer.writerow(row)
    pprint.pprint(row)


def main():
    parser = argparse.ArgumentParser(
        description="Logistic DDOS detector with Keras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t','--train',       nargs=1,      help='Start training (folder)')
    parser.add_argument('-e','--epochs',      type=int,      default=100, help='Epochs')
    parser.add_argument('-cv','--cross_validation', type=int, default=0,   help='Ignored for logistic')
    parser.add_argument('-p','--predict',     type=str,      help='Folder with *test.hdf5')
    parser.add_argument('-i','--iterations',  type=int,      default=1,   help='Avg this many predict runs')
    parser.add_argument('-m','--model',       type=str,      help='.h5 model file to load')
    parser.add_argument('-pl','--predict_live', type=str, help='Interface or pcap file for live prediction')

    args = parser.parse_args()

    # -------- TRAIN --------
    if args.train:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        train_folder = args.train[0]
        subfolders = glob.glob(train_folder.rstrip('/') + "/*/")
        if not subfolders:
            subfolders = [train_folder + "/"]
        for full_path in sorted(subfolders):
            ds = full_path.rstrip('/')
            train_file = glob.glob(f"{ds}/*-train.hdf5")[0]
            val_file   = glob.glob(f"{ds}/*-val.hdf5")[0]

            X_train, Y_train = load_hdf5_dataset(train_file)
            X_val,   Y_val   = load_hdf5_dataset(val_file)
            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val,   Y_val   = shuffle(X_val,   Y_val,   random_state=SEED)

            input_dim  = X_train.shape[1]
            model_name = ds.split("/")[-1] + "-logistic"
            model      = build_logistic_model(input_dim)

            ckpt_path  = f"{OUTPUT_FOLDER}/{model_name}.h5"
            checkpoint = ModelCheckpoint(
                ckpt_path,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1
            )
            # stop as soon as val_accuracy stops improving
            early = EarlyStopping(
                monitor="val_accuracy",
                patience=0,
                verbose=1
            )

            # TRAIN
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=args.epochs,
                batch_size=32,
                callbacks=[checkpoint, early],
                verbose=1
            )

            # load the best‐saved weights
            best = load_model(ckpt_path)
            Y_pred = (best.predict(X_val) > 0.5).astype(int).ravel()
            Y_true = Y_val.ravel()

            # final validation accuracy on its own line
            acc = accuracy_score(Y_true, Y_pred)
            print(f"{acc:.4f}")

            # print the best hyper-parameters
            print("Best parameters: ", {})  
            print("Best model path: ", ckpt_path)

            # F1 and confusion matrix
            f1_val = f1_score(Y_true, Y_pred)
            cm = confusion_matrix(Y_true, Y_pred)
            print("F1 Score of the best model on the validation set: ", f1_val)
            print("Confusion Matrix: ", cm)

            csv_path = f"{OUTPUT_FOLDER}/{model_name}.csv"
            with open(csv_path, 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=VAL_HEADER)
                wr.writeheader()
                wr.writerow({
                  'Model': model_name,
                  'Samples': Y_pred.shape[0],
                  'Accuracy': f"{acc:0.4f}",
                  'F1Score':  f"{f1_val:0.4f}",
                  'Hyper-parameters': {},
                  'Validation Set': val_file
                })

            logging.info("Done %s: acc=%.4f f1=%.4f csv=%s",
                         model_name, acc, f1_val, csv_path)


    # -------- PREDICT (batch) --------
    if args.predict:
        if not args.model:
            print("Error: must specify --model when using --predict", file=sys.stderr)
            sys.exit(1)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        predict_files = glob.glob(f"{args.predict}/*test.hdf5")
        out_csv = f"{OUTPUT_FOLDER}/predictions-{int(time.time())}.csv"
        with open(out_csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=PREDICT_HEADER)
            wr.writeheader()
            for ds in predict_files:
                X_test, Y_test = load_hdf5_dataset(ds)
                model = load_model(args.model)

                total_time = 0.0
                for _ in range(args.iterations):
                    t0 = time.time()
                    Y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
                    total_time += time.time() - t0
                avg_time = total_time / args.iterations

                report_results(Y_test, Y_pred, avg_time, model.name, ds, wr)
        logging.info("Predictions written to %s", out_csv)

    # -------- PREDICT LIVE --------
    if args.predict_live:
        if not args.model:
            print("Error: must specify --model when using --predict_live", file=sys.stderr)
            sys.exit(1)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        live_csv = f"{OUTPUT_FOLDER}/predictions-live-{int(time.time())}.csv"
        predict_file = open(live_csv, 'w', newline='')
        writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        writer.writeheader()
        predict_file.flush()

        if args.predict_live.endswith('.pcap'):
            cap = pyshark.FileCapture(args.predict_live)
            data_source = os.path.basename(args.predict_live)
        else:
            cap = pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print("Prediction on network traffic from:", data_source)

        labels    = parse_labels(None, None, None)
        model     = load_model(args.model)
        input_dim = model.input_shape[1]
        tw        = TIME_WINDOW
        mins, maxs = static_min_max(tw)

        while True:
            samples = process_live_traffic(
                cap, None, labels,
                max_flow_len=input_dim,
                traffic_type="all",
                time_window=tw
            )
            if not samples and isinstance(cap, pyshark.FileCapture):
                print("\nNo more packets in file", data_source)
                break

            if samples:
                # now unpack keys to get ip_src
                X_list, Y_true, keys = dataset_to_list_of_fragments(samples)
                X_padded = normalize_and_padding(X_list, mins, maxs, input_dim)
                X = np.stack([sample[:, 0] for sample in X_padded], axis=0)

                t0 = time.time()
                Y_pred = (model.predict(X, batch_size=2048) > 0.5).astype(int).ravel()
                t1 = time.time()
                prediction_time = t1 - t0

                # pull IPs from keys
                ip_addresses = [keys[i][0] for i, pred in enumerate(Y_pred) if pred]
                print("Predicted IP addresses:", ip_addresses)

                report_results(Y_true, Y_pred, prediction_time, model.name, data_source, writer)
                predict_file.flush()

        predict_file.close()


if __name__ == "__main__":
    main()
