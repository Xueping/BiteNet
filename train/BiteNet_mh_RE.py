from utils.configs import cfg
from utils.record_log import RecordLog
import numpy as np
from BiteNet.model_mh import BiteNet as Model
import os
from tensorflow import keras
from dataset.dataset_full import VisitDataset
import warnings
import tensorflow as tf
from utils.evaluation import ConceptEvaluation as CodeEval, \
    EvaluationTemplate as Evaluation

warnings.filterwarnings('ignore')
logging = RecordLog()


def train():

    visit_threshold = cfg.visit_threshold
    epochs = cfg.max_epoch
    batch_size = cfg.train_batch_size

    data_set = VisitDataset()
    data_set.prepare_data(visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    code_eval = CodeEval(data_set, logging)
    print(data_set.train_context_codes.shape)
    print(data_set.train_intervals.shape)
    print(data_set.train_labels_2.shape)

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    # es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    model = Model(data_set)
    model.build_network()
    model.model.fit(x=[data_set.train_context_codes,data_set.train_intervals],
                    y=data_set.train_labels_2,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([data_set.dev_context_codes,data_set.dev_intervals], data_set.dev_labels_2)
                    # , callbacks=[es]
                    )

    metrics = model.model.evaluate([data_set.test_context_codes, data_set.test_intervals], data_set.test_labels_2)
    log_str = 'Single fold accuracy is {}'.format(metrics[1])
    logging.add(log_str)

    predicts = model.model.predict([data_set.test_context_codes, data_set.test_intervals])
    predict_classes = predicts > 0.5
    predict_classes = predict_classes.astype(np.int)
    metrics = Evaluation.metric_pred(data_set.test_labels_2, predicts, predict_classes)
    logging.add(metrics)
    logging.done()

    embedding_weights = model.embedding.shared_weights
    embedding_values = tf.keras.backend.get_value(embedding_weights)

    icd__nmi = code_eval.get_clustering_nmi(embedding_values, 'ICD')
    logging.add('ICD, NMI Score: ' + str(icd__nmi))
    ccs__nmi = code_eval.get_clustering_nmi(embedding_values, 'CCS')
    logging.add('CCS, NMI Score: ' + str(ccs__nmi))

    for k in [1, 5, 10]:
        icd__nns = code_eval.get_nns_p_at_top_k(embedding_values, 'ICD', k)
        logging.add('ICD, nns Score: ' + str(icd__nns))
        ccs__nns = code_eval.get_nns_p_at_top_k(embedding_values, 'CCS', k)
        logging.add('CCS, nns Score: ' + str(ccs__nns))


def test():
    pass


def main():
    if cfg.train:
        train()
    else:
        test()


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    main()