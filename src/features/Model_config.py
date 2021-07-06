class ModelConfig(object):
  def __init__(self, max_epochs, batch_size, batch_size_test, freq_of_test,
               learning_rate, lr_scheduler_step_size, lr_decay,
               per_series_lr_multip, gradient_eps, 
              #  gradient_clipping_threshold,
               transformer_weight_decay,
               noise_std,
               level_variability_penalty,
               testing_percentile, training_percentile, ensemble,              
               seasonality, input_size, output_size, 
               frequency, max_periods, random_seed, device, root_dir,
               d_input, d_model, d_output, q, v, h, N, attention_size, dropout, chunk_mode, pe, pe_period):

    # Train Parameters
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.batch_size_test = batch_size_test
    self.freq_of_test = freq_of_test
    self.learning_rate = learning_rate
    self.lr_scheduler_step_size = lr_scheduler_step_size
    self.lr_decay = lr_decay
    self.per_series_lr_multip = per_series_lr_multip
    self.gradient_eps = gradient_eps
    # self.gradient_clipping_threshold = gradient_clipping_threshold
    self.transformer_weight_decay = transformer_weight_decay
    self.noise_std = noise_std
    self.level_variability_penalty = level_variability_penalty
    self.testing_percentile = testing_percentile
    self.training_percentile = training_percentile
    self.ensemble = ensemble
    self.device = device

    # Model Parameters
    self.random_seed = random_seed
    # Transformer parameters
    self.d_input = d_input
    self.d_model = d_model
    self.d_output = d_output
    self.q = q
    self.v = v
    self.h = h
    self.N = N
    self.attention_size = attention_size
    self.dropout = dropout
    self.chunk_mode = chunk_mode
    self.pe = pe
    self.pe_period = pe_period

    # Data Parameters
    self.seasonality = seasonality
    if len(seasonality)>0:
      self.naive_seasonality = seasonality[0]
    else:
      self.naive_seasonality = 1
    self.input_size = input_size
    self.input_size_i = self.input_size
    self.output_size = output_size
    self.output_size_i = self.output_size
    self.frequency = frequency
    self.min_series_length = self.input_size_i + self.output_size_i
    self.max_series_length = (max_periods * self.input_size) + self.min_series_length
    self.max_periods = max_periods
    self.root_dir = root_dir