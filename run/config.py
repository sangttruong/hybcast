# Inital configuration
from yacs.config import CfgNode as CN


cfg = CN()

cfg.device = "cpu"

cfg.dataset = CN()
cfg.dataset.name = None
cfg.dataset.input_directory = None
cfg.dataset.num_obs = None


cfg.train_parameters = CN()
cfg.train_parameters.max_epochs = None
cfg.train_parameters.batch_size = None
cfg.train_parameters.freq_of_test = None
cfg.train_parameters.learning_rate = None
cfg.train_parameters.per_series_lr_multip = None
cfg.train_parameters.lr_scheduler_step_size = None
cfg.train_parameters.lr_decay = None
cfg.train_parameters.level_variability_penalty = None
cfg.train_parameters.testing_percentile = None
cfg.train_parameters.training_percentile = None
cfg.train_parameters.ensemble = False

cfg.ES_parameters = CN()
cfg.ES_parameters.max_periods = None
cfg.ES_parameters.seasonality = None
cfg.ES_parameters.input_size = None
cfg.ES_parameters.output_size = None
cfg.ES_parameters.d_input = None
cfg.ES_parameters.d_output = None
cfg.ES_parameters.random_seed = None

cfg.Transformer_parameter = CN()
cfg.Transformer_parameter.d_model = None
cfg.Transformer_parameter.q = None
cfg.Transformer_parameter.v = None
cfg.Transformer_parameter.h = None
cfg.Transformer_parameter.N = None
cfg.Transformer_parameter.attention_size = None
cfg.Transformer_parameter.dropout = None
cfg.Transformer_parameter.chunk_mode = None 
cfg.Transformer_parameter.pe = None 
cfg.Transformer_parameter.pe_period = None
