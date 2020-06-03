import os
import numpy as np

class LibPathConfig():

    def __init__(self):
        self.__formalized_dict_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                                     "id_preprocessing/resources/formalizationDict.csv")

        self.__stopword_list_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                            "id_preprocessing/resources/stopword.csv")
        
        self.__vocab_list_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                            "id_preprocessing/resources/id_kateglo.vocab")

    def get_formalized_dict_path(self):
        return self.__formalized_dict_path__

    def get_stopword_list_path(self):
        return self.__stopword_list_path__
    
    def get_vocab_list_path(self):
        return self.__vocab_list_path__

class CRFTokenizerConfig():

    def __init__(self):
        self.__default_model_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                                   "util/tokenization/id/model/crf_tokenizer.pkl")
        """
        Default CRF paramter
        algorithm               : Training algorithm, defaulted to lbfgs,
                                    Gradient descent using the L-BFGS method.
        c1                      : float, Coefficient for L1 Normalization.
        c2                      : float, Coefficient for L2 Normalization.
        max_iterations          : integer, Maximum number of training iteration.
        all_possible_transitions: bool, A flag whether to use every possible label transition,
                                        even the one that didn't appear on traning data.
        """
        self.__default_train_param__ = {
                                        "algorithm":'lbfgs',
                                        "c1":1.2,
                                        "c2":0.0,
                                        "max_iterations":300,
                                        "all_possible_transitions":True
                                    }
        self.__tuning_grid__ = {
            "c1": np.random.choice(np.arange(0., 1.5, 0.1), size=10, replace=False),
            "c2": np.random.choice(np.arange(0., 1.5, 0.01), size=10, replace=False),
            "max_iterations": np.arange(100, 501, 100)
        }
        """
        Grid search settings
        cv      : integer, number of K in K-fold
        n_jobs  : integer, number of processor core to be utilized, -1 means use all
        verbose : integer, verbose flag
        """
        self.__gs_param__ = {
            "cv":5,
            "n_jobs":-1, 
            "verbose":1
        }

    def get_default_model(self):
        return self.__default_model_path__
    
    def get_default_train_param(self):
        return self.__default_train_param__

    def get_default_tuning_grid(self):
        return self.__tuning_grid__

    def get_default_gs_setting(self):
        return self.__gs_param__

class CRFLangaugePredictorConfig():
    def __init__(self):
        self.__default_model_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                                   "util/language_detection/model/all_pretrained_crf.zip")
        """
        Default Feature Parameter
        ngram   : Integer, order of character ngram of a word to be used as a feature
        window  : Integer, number of tokens before and after current one to be used as feature. 
        """
        self.__default_feature_param__ = {
            "ngram":3,
            "window":5,
            "everygram":False
        }
        """
        Default CRF paramter
        algorithm               : Training algorithm, defaulted to lbfgs,
                                    Gradient descent using the L-BFGS method.
        c1                      : float, Coefficient for L1 Normalization.
        c2                      : float, Coefficient for L2 Normalization.
        max_iterations          : integer, Maximum number of training iteration.
        all_possible_transitions: bool, A flag whether to use every possible label transition,
                                        even the one that didn't appear on traning data.
        """
        self.__default_train_param__ = {
            "algorithm":'lbfgs',
            "c1":0.1,
            "c2":0.02,
            "max_iterations":200,
            "all_possible_transitions":True
        }

        self.__tuning_grid__ = {
            "c1": np.random.choice(np.arange(0., 1.5, 0.1), size=10, replace=False),
            "c2": np.random.choice(np.arange(0., 1.5, 0.01), size=10, replace=False),
            "max_iterations": np.arange(100, 501, 100)
        }
        """
        Grid search settings
        cv      : integer, number of K in K-fold
        n_jobs  : integer, number of processor core to be utilized, -1 means use all
        verbose : integer, verbose flag
        """
        self.__gs_param__ = {
            "cv":5,
            "n_jobs":-1, 
            "verbose":1
        }
    
    def get_default_model(self):
        return self.__default_model_path__
    
    def get_default_feature_param(self):
        return self.__default_feature_param__

    def get_default_train_param(self):
        return self.__default_train_param__
    
    def get_default_tuning_grid(self):
        return self.__tuning_grid__

    def get_default_gs_setting(self):
        return self.__gs_param__

class CNNLanguangePredictorConfig():
    def __init__(self):
        self.__default_model_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                                    "util/language_detection/model/all_pretrained_cnn.zip")
        
        """
        :params input_size      : integer indicates Convolution input length (word length) 
        :params optimizer       : string or one of keras.optimizer class instances
        :params loss            : string or one of keras.losses class instances
        :params dropout         : float indicates dropout probability 0.1-0.5 range is recommended,
                                    Dropout will be used after each FCN layer
        :params conv_layers     : list of list layer settings, each layer setting is list with length 3
                                    indicating Number of filter, filter size, and max pooling
                                    (optional, set to -1 if no pooling)
        :params fcn_layers      : list of integer indicating number of unit of each layer
        """
        self.__default_cnn_params__ = {
            "input_size":50,
            "optimizer":"adam",
            "loss":"categorical_crossentropy",
            "dropout": 0.5,
            "conv_layers":[[256, 5, 2], [258, 3, -1]],
            "fcn_layers":[512,256],
            "crf_context_size": 1,
            "lower": False
        }
        
        """
        Default CRF paramter
        algorithm               : Training algorithm, defaulted to lbfgs,
                                    Gradient descent using the L-BFGS method.
        c1                      : float, Coefficient for L1 Normalization.
        c2                      : float, Coefficient for L2 Normalization.
        max_iterations          : integer, Maximum number of training iteration.
        all_possible_transitions: bool, A flag whether to use every possible label transition,
                                        even the one that didn't appear on traning data.
        """
        self.__default_crf_param__ = {
            "algorithm":'lbfgs',
            "c1":0.1,
            "c2":0.1,
            "max_iterations":300,
            "all_possible_transitions":True
        }

        self.__default_tuning_grid__ = {
            "cnn_param":[
                [[64, 3, 2]],
                [   [64, 5, 1],
                    [128, 3, 1]
                ],
                [   [128, 7, 3],
                    [256, 5, 2]
                ],
                [   [256, 5, 3],
                    [256, 3, 2]
                ]
            ],
            "fcn_param":[
                [512, 256],
                [1024, 256], 
                [512],
                [256]
            ],
            "drop_out_params":np.arange(0.1, 0.51, 0.1),
            "lr_params":np.arange(0.001, 0.011, 0.002)
        }

    def get_default_model(self):
        return self.__default_model_path__
    
    def get_cnn_params(self):
        return self.__default_cnn_params__
    
    def get_crf_params(self):
        return self.__default_cnn_params__
    
    def get_default_tuning_grid(self):
        return self.__tuning_grid__


class RNNLanguangePredictorConfig():
    def __init__(self):
        self.__default_model_path__ = os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],
                                                    "util/language_detection/model/all_pretrained_rnn.zip")
        
        """
        :params input_size      : integer indicates Convolution input length (word length) 
        :params optimizer       : string or one of keras.optimizer class instances
        :params loss            : string or one of keras.losses class instances
        :params dropout         : float indicates dropout probability 0.1-0.5 range is recommended,
                                    Dropout will be used after each FCN layer
        :params conv_layers     : list of list layer settings, each layer setting is list with length 3
                                    indicating Number of filter, filter size, and max pooling
                                    (optional, set to -1 if no pooling)
        :params fcn_layers      : list of integer indicating number of unit of each layer
        """
        self.__default_rnn_params__ = {
            "seq_length":100, 
            "rnn_units":125,
            "learning_rate":0.005,
            "recurrent_dropout":0.2, 
            "embedding_dropout":0.2,
            "embed_type":"word", 
            "vocab_size":0,
        }
        self.__default_cnn_params__ = {
            "conv_layers":[[128, 5, 2],[256, 3, -1]],
            "word_length":50,
            "char_embed_size":100, 
        }

    def get_default_model(self):
        return self.__default_model_path__
    
    def get_cnn_params(self):
        return self.__default_cnn_params__
    
    def get_rnn_params(self):
        return self.__default_rnn_params__

class SlangTranslationConfig():
    def __init__(self):
        self.__default_model_path__ = {
            "word":os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],"util/smt/smt_pretrained_moses.zip"),
            "char":os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],"util/smt/char_smt_pretrained_moses.zip")
        }

        self.__default_directory_path__ = {
            "train_dir":os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],"util/smt/smt-model"),
            "tmp_prefix":"/tmp/", 
            "moses_dir":os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],"util/smt/smt-core/mosesdecoder"),
            "external_tools_dir":os.path.join(os.environ["LANGUAGE_ENGINE_PATH"],"smt-core/external-bin-dir"),
        }

    def get_default_model(self):
        return self.__default_model_path__
    
    def get_default_directory(self):
        return self.__default_directory_path__