import datetime
import json
import os.path as osp
import pickle

import numpy as np
import torch

from base.train_model import test, train
from base.utils import Averager, ensure_path, get_metrics


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        ensure_path(self.args.result_dir)
        self.text_file = self.args.results_file
        self.subject_results = []
        with open(self.text_file, "a", encoding="utf-8") as file:
            file.write(
                "\n{}\nTrain:Parameter setting for {} on {}:\n".format(
                    datetime.datetime.now(), args.model, args.dataset
                )
            )
            for idx, key in enumerate(sorted(args.__dict__.keys())):
                file.write("{}){}:{};".format(idx, key, args.__dict__[key]))
            file.write("\n")

    def load_per_subject(self, sub):
        save_path = osp.join(self.args.ROOT, "data_processed")
        data_type = "data_{}_{}_{}".format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = "sub" + str(sub) + ".pkl"
        path = osp.join(save_path, data_type, sub_code)
        with open(path, "rb") as file:
            dataset = pickle.load(file)
        return dataset["data"], dataset["label"]

    def prepare_data(self, data_train, label_train, data_test, label_test):
        data_train, label_train = np.concatenate(data_train), np.concatenate(label_train)
        data_test, label_test = np.concatenate(data_test), np.concatenate(label_test)
        data_train, data_test = self.normalize_channel_wise(data_train, data_test)
        data_train, label_train = self.from_numpy_to_tensor(data_train, label_train)
        data_test, label_test = self.from_numpy_to_tensor(data_test, label_test)
        return data_train, label_train, data_test, label_test

    def normalize_channel_wise(self, train, test):
        if train.ndim > 4:
            temp_train = np.reshape(
                train,
                (train.shape[0], train.shape[1], train.shape[2], train.shape[3] * train.shape[4]),
            )
            temp_test = np.reshape(
                test,
                (test.shape[0], test.shape[1], test.shape[2], test.shape[3] * test.shape[4]),
            )

            for channel in range(temp_train.shape[-1]):
                mean = np.mean(temp_train[:, :, :, channel])
                std = np.std(temp_train[:, :, :, channel])
                if std != 0:
                    temp_train[:, :, :, channel] = (temp_train[:, :, :, channel] - mean) / std
                    temp_test[:, :, :, channel] = (temp_test[:, :, :, channel] - mean) / std
                train = np.reshape(temp_train, train.shape)
                test = np.reshape(temp_test, test.shape)

        elif train.ndim == 4:
            if self.args.data_format == "PSD_DE":
                for channel in range(train.shape[-2]):
                    mean_psd = np.mean(train[:, :, channel, :7])
                    std_psd = np.std(train[:, :, channel, :7])
                    mean_de = np.mean(train[:, :, channel, 7:])
                    std_de = np.std(train[:, :, channel, 7:])

                    train[:, :, channel, :7] = (train[:, :, channel, :7] - mean_psd) / std_psd
                    test[:, :, channel, :7] = (test[:, :, channel, :7] - mean_psd) / std_psd
                    train[:, :, channel, 7:] = (train[:, :, channel, 7:] - mean_de) / std_de
                    test[:, :, channel, 7:] = (test[:, :, channel, 7:] - mean_de) / std_de
            else:
                for channel in range(train.shape[-2]):
                    mean = np.mean(train[:, :, channel, :])
                    std = np.std(train[:, :, channel, :])
                    train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
                    test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std

        else:
            for channel in range(train.shape[-2]):
                mean = np.mean(train[:, channel, :])
                std = np.std(train[:, channel, :])
                train[:, channel, :] = (train[:, channel, :] - mean) / std
                test[:, channel, :] = (test[:, channel, :] - mean) / std

        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        np.random.seed(0)
        num_class = int(max(label)) + 1
        index = []
        for class_id in range(num_class):
            idx_this_class = np.where(label == class_id)[0]
            if random:
                np.random.shuffle(idx_this_class)
            index.append(idx_this_class)

        idx_train, idx_val = [], []
        for idx in index:
            idx_train.extend(idx[: int(len(idx) * train_rate)])
            idx_val.extend(idx[int(len(idx) * train_rate) :])

        return data[idx_train], label[idx_train], data[idx_val], label[idx_val]

    @staticmethod
    def from_numpy_to_tensor(data, label):
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

    def leave_sub_out(self, subject=None, shuffle=True, reproduce=False):
        if subject is None:
            subject = []

        tta = []
        ttf = []
        tva = []

        for sub in subject:
            va_val = Averager()
            preds, acts = [], []
            data_train, label_train = [], []
            data_test, label_test = self.load_per_subject(sub)
            for sub_ in subject:
                if sub != sub_:
                    data_temp, label_temp = self.load_per_subject(sub_)
                    data_train.extend(data_temp)
                    label_train.extend(label_temp)

            data_train, label_train, data_test, label_test = self.prepare_data(
                data_train=data_train,
                label_train=label_train,
                data_test=data_test,
                label_test=label_test,
            )
            print("Training:{}  Test: {}".format(data_train.size(), data_test.size()))
            if reproduce:
                acc_test, pred, act = test(
                    args=self.args,
                    data=data_test,
                    label=label_test,
                    reproduce=self.args.reproduce,
                    subject=sub,
                    trial=0,
                )
                acc_val = 0
            else:
                data_train, label_train, data_val, label_val = self.split_balance_class(
                    data=data_train,
                    label=label_train,
                    train_rate=0.8,
                    random=shuffle,
                )

                acc_val = train(
                    args=self.args,
                    data_train=data_train,
                    label_train=label_train,
                    data_val=data_val,
                    label_val=label_val,
                    subject=sub,
                    trial=0,
                )

                acc_test, pred, act = test(
                    args=self.args,
                    data=data_test,
                    label=label_test,
                    reproduce=self.args.reproduce,
                    subject=sub,
                    trial=0,
                )

            va_val.add(acc_val)
            preds.extend(pred)
            acts.extend(act)

            tva.append(va_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)

            fold_result = {
                "subject": int(sub),
                "test_acc": float(acc_test),
                "agg_acc": float(acc),
                "agg_f1": float(f1),
                "val_acc": float(va_val.item()),
            }
            self.subject_results.append(fold_result)
            self.log2txt(json.dumps(fold_result, ensure_ascii=False))

        summary = {
            "dataset": self.args.dataset,
            "experiment_name": self.args.experiment_name,
            "result_dir": self.args.result_dir,
            "mean_test_acc": float(np.mean(tta)),
            "std_test_acc": float(np.std(tta)),
            "mean_test_f1": float(np.mean(ttf)),
            "std_test_f1": float(np.std(ttf)),
            "mean_val_acc": float(np.mean(tva)),
            "subjects": self.subject_results,
        }
        results = "Test mAcc={}({}) mF1={}({}) Val mAcc={}".format(
            summary["mean_test_acc"],
            summary["std_test_acc"],
            summary["mean_test_f1"],
            summary["std_test_f1"],
            summary["mean_val_acc"],
        )
        print(results)
        self.log2txt(results)
        with open(self.args.summary_file, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    def log2txt(self, content):
        with open(self.text_file, "a", encoding="utf-8") as file:
            file.write(str(content) + "\n")
