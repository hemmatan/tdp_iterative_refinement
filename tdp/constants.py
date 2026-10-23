
MAXLEN_BERT = 510
STRIDE_BERT = 350
APPLY_ALGEBRA = False
APP_START_TIMESTAMP = "TIMESTAMP_NOT_SET"
PROJECT_PATH = "PATH_NOT_SET"
CURRENT_RUN_LOG_DIR = "DIR_NOT_SET"
LOGGER = None
INF = 1000000000000.0
TERMINATION_MESSAGE = "Application terminated."


METRICS = {
    "val_f1_macro_labeled_at_maxvalf1": 0,
    "val_f1_macro_unlabeled_at_maxvalf1": 0,
    "val_f1_micro_labeled_at_maxvalf1": 0,
    "val_f1_micro_unlabeled_at_maxvalf1": 0,
    "validation_f1_micro_labeled_at_minloss": 0,
    "validation_f1_macro_labeled_at_minloss": 0,
    "validation_f1_micro_unlabeled_at_minloss": 0,
    "validation_f1_macro_unlabeled_at_minloss": 0,
    "train_f1_micro_labeled_at_minloss": 0,
    "train_f1_macro_labeled_at_minloss": 0,
    "train_f1_micro_unlabeled_at_minloss": 0,
    "train_f1_macro_unlabeled_at_minloss": 0,
    "train_f1_micro_labeled_at_maxvalf1": 0,
    "train_f1_macro_labeled_at_maxvalf1": 0,
    "train_f1_micro_unlabeled_at_maxvalf1": 0,
    "train_f1_macro_unlabeled_at_maxvalf1": 0,
    "test_f1_macro_labeled_at_maxvalf1": 0,
    "test_f1_macro_unlabeled_at_maxvalf1": 0,
    "epoch_min_loss": 0,
    "epoch_max_val_f1_macro_labeled": 0,
    "val_precision_macro_labeled_at_maxvalf1": 0,
    "val_precision_macro_unlabeled_at_maxvalf1": 0,
    "val_precision_micro_labeled_at_maxvalf1": 0,
    "val_precision_micro_unlabeled_at_maxvalf1": 0,
    "val_recall_macro_labeled_at_maxvalf1": 0,
    "val_recall_macro_unlabeled_at_maxvalf1": 0,
    "val_recall_micro_labeled_at_maxvalf1": 0,
    "val_recall_micro_unlabeled_at_maxvalf1": 0,
    "train_precision_macro_labeled_at_maxvalf1": 0,
    "train_precision_macro_unlabeled_at_maxvalf1": 0,
    "train_precision_micro_labeled_at_maxvalf1": 0,
    "train_precision_micro_unlabeled_at_maxvalf1": 0,
    "train_recall_macro_labeled_at_maxvalf1": 0,
    "train_recall_macro_unlabeled_at_maxvalf1": 0,
    "train_recall_micro_labeled_at_maxvalf1": 0,
    "train_recall_micro_unlabeled_at_maxvalf1": 0,
    "validation_precision_micro_labeled_at_minloss": 0,
    "validation_precision_macro_labeled_at_minloss": 0,
    "validation_precision_micro_unlabeled_at_minloss": 0,
    "validation_precision_macro_unlabeled_at_minloss": 0,
    "validation_recall_micro_labeled_at_minloss": 0,
    "validation_recall_macro_labeled_at_minloss": 0,
    "validation_recall_micro_unlabeled_at_minloss": 0,
    "validation_recall_macro_unlabeled_at_minloss": 0,
    "train_precision_micro_labeled_at_minloss": 0,
    "train_precision_macro_labeled_at_minloss": 0,
    "train_precision_micro_unlabeled_at_minloss": 0,
    "train_precision_macro_unlabeled_at_minloss": 0,
    "train_recall_micro_labeled_at_minloss": 0,
    "train_recall_macro_labeled_at_minloss": 0,
    "train_recall_micro_unlabeled_at_minloss": 0,
    "train_recall_macro_unlabeled_at_minloss": 0,
}


class METRIC_NAMES:
    unlabeled_tp = "unlabeled_tp"
    unlabeled_fp = "unlabeled_fp"
    unlabeled_fn = "unlabeled_fn"
    unlabeled_f1_micro = "unlabeled_f1_micro"
    unlabeled_f1_macro = "unlabeled_f1_macro"
    labeled_f1_micro = "labeled_f1_micro"
    labeled_f1_macro = "labeled_f1_macro"
    labeled_f1_doc_macro = "labeled_f1_doc_macro"
    unlabeled_f1_doc_macro = "unlabeled_f1_doc_macro"
    unlabeled_confusion_matrix = "unlabeled_confusion_matrix"
    labeled_confusion_matrix = "labeled_confusion_matrix"
    labeled_f1_scores = "labeled_f1_scores"
    weighted_average_f1 = "weighted_average_f1"
    unlabeled_f1 = "unlabeled_f1"
    labeled_f1 = "labeled_f1"
    confusion = "confusion_matrix_labeled"
    confusion_unlabeled = "confusion_matrix_unlabeled"
    labeled_precision_micro = "labeled_precision_micro"
    labeled_precision_macro = "labeled_precision_macro"
    labeled_recall_micro = "labeled_recall_micro"
    labeled_recall_macro = "labeled_recall_macro"
    unlabeled_precision_micro = "unlabeled_precision_micro"
    unlabeled_precision_macro = "unlabeled_precision_macro"
    unlabeled_precision_doc_macro = "unlabeled_precision_doc_macro"
    unlabeled_recall_micro = "unlabeled_recall_micro"
    unlabeled_recall_macro = "unlabeled_recall_macro"
    unlabeled_recall_doc_macro = "unlabeled_recall_doc_macro"
    labeled_f1_micro_all = "labeled_f1_micro_all"
    labeled_f1_macro_all = "labeled_f1_macro_all"
    labeled_f1_doc_macro_all = "labeled_f1_doc_macro_all"
    labeled_precision_micro_all = "labeled_precision_micro_all"
    labeled_precision_macro_all = "labeled_precision_macro_all"
    labeled_recall_micro_all = "labeled_recall_micro_all"
    labeled_recall_macro_all = "labeled_recall_macro_all"
    labeled_f1_micro_temporal = "labeled_f1_micro_temporal"
    labeled_f1_macro_temporal = "labeled_f1_macro_temporal"
    labeled_f1_doc_macro_temporal = "labeled_f1_doc_macro_temporal"
    labeled_precision_micro_temporal = "labeled_precision_micro_temporal"
    labeled_precision_macro_temporal = "labeled_precision_macro_temporal"
    labeled_recall_micro_temporal = "labeled_recall_micro_temporal"
    labeled_recall_macro_temporal = "labeled_recall_macro_temporal"
    unlabeled_f1_micro_all = "unlabeled_f1_micro_all"
    unlabeled_f1_macro_all = "unlabeled_f1_macro_all"
    unlabeled_f1_doc_macro_all = "unlabeled_f1_doc_macro_all"
    unlabeled_precision_micro_all = "unlabeled_precision_micro_all"
    unlabeled_precision_macro_all = "unlabeled_precision_macro_all"
    unlabeled_precision_doc_macro_all = "unlabeled_precision_doc_macro_all"
    unlabeled_recall_micro_all = "unlabeled_recall_micro_all"
    unlabeled_recall_macro_all = "unlabeled_recall_macro_all"
    unlabeled_recall_doc_macro_all = "unlabeled_recall_doc_macro_all"
    unlabeled_f1_micro_temporal = "unlabeled_f1_micro_temporal"
    unlabeled_f1_macro_temporal = "unlabeled_f1_macro_temporal"
    unlabeled_f1_doc_macro_temporal = "unlabeled_f1_doc_macro_temporal"
    unlabeled_precision_micro_temporal = "unlabeled_precision_micro_temporal"
    unlabeled_precision_macro_temporal = "unlabeled_precision_macro_temporal"
    unlabeled_precision_doc_macro_temporal = "unlabeled_precision_doc_macro_temporal"
    unlabeled_recall_micro_temporal = "unlabeled_recall_micro_temporal"
    unlabeled_recall_macro_temporal = "unlabeled_recall_macro_temporal"
    unlabeled_recall_doc_macro_temporal = "unlabeled_recall_doc_macro_temporal"
