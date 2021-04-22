import os, time, torch
import numpy as np
import pandas as pd

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from copy import deepcopy

# Utils
from Utils.Model_config import *
from Utils.Loss import *
from Utils.Batch import * 
from Utils.Comparison import *

from ESTransformer import ESTransformer

class DeployedESTransformer(object):
  def __init__(self, max_epochs=15, batch_size=1, batch_size_test=64, freq_of_test=-1,
               learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
               per_series_lr_multip=1.0, gradient_eps=1e-8,
               transformer_weight_decay=0, noise_std=0.001,
               level_variability_penalty=80,
               testing_percentile=50, training_percentile=50, ensemble=False,
               
               seasonality=[4],
               input_size=4,
               output_size=8,
               frequency=None, max_periods=20, random_seed=1,
               device = 'cpu', root_dir= './',

               # Transformer parameters
               d_input = 4,
               d_model = 48,
               d_output = 6,
               q = 8,
               v = 8,
               h = 4,
               N = 4,
               attention_size = None,
               dropout = 0.3,
               chunk_mode = 'chunk',
               pe = None,
               pe_period = 24):
    
    super().__init__()
    self.mc = ModelConfig(max_epochs=max_epochs, batch_size=batch_size, batch_size_test=batch_size_test,
                          freq_of_test=freq_of_test, learning_rate=learning_rate,
                          lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
                          per_series_lr_multip=per_series_lr_multip,
                          gradient_eps=gradient_eps,
                          transformer_weight_decay=transformer_weight_decay, noise_std=noise_std,
                          level_variability_penalty=level_variability_penalty,
                          testing_percentile=testing_percentile, training_percentile=training_percentile,
                          ensemble=ensemble,

                          seasonality=seasonality,
                          input_size=input_size,
                          
                          output_size=output_size,
                          frequency=frequency, max_periods=max_periods, random_seed=random_seed,
                          device=device, root_dir=root_dir,

                          d_input = d_input,
                          d_model = d_model,
                          d_output = d_output,
                          q = q,
                          v = v,
                          h = h,
                          N = N,
                          attention_size = attention_size,
                          dropout = dropout,
                          chunk_mode = chunk_mode,
                          pe = pe,
                          pe_period = pe_period)
    self._fitted = False

  def instantiate_estransformer(self, exogenous_size, n_series):    
    self.mc.exogenous_size = exogenous_size
    self.mc.n_series = n_series
    self.estransformer = ESTransformer(self.mc).to(self.mc.device)

  def fit(self, X_df, y_df, X_test_df=None, y_test_df=None, y_hat_benchmark='y_hat_naive2',
          warm_start=False, shuffle=True, verbose=True):
    # Transform long dfs to wide numpy
    assert type(X_df) == pd.core.frame.DataFrame
    assert type(y_df) == pd.core.frame.DataFrame
    assert all([(col in X_df) for col in ['unique_id', 'ds', 'x']])
    assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])
    if y_test_df is not None:
      assert y_hat_benchmark in y_test_df.columns, 'benchmark is not present in y_test_df, use y_hat_benchmark to define it'

    # Storing dfs for OWA evaluation, initializing min_owa
    self.y_train_df = y_df
    self.X_test_df = X_test_df
    self.y_test_df = y_test_df
    self.min_owa = 4.0
    self.min_epoch = 0

    self.int_ds = isinstance(self.y_train_df['ds'][0], (int, np.int, np.int64))

    self.y_hat_benchmark = y_hat_benchmark

    X, y = self.long_to_wide(X_df, y_df)
    assert len(X)==len(y)
    assert X.shape[1]>=3

    # Exogenous variables
    unique_categories = np.unique(X[:, 1])
    self.mc.category_to_idx = dict((word, index) for index, word in enumerate(unique_categories))
    exogenous_size = len(unique_categories)

    # Create batches (device in mc)
    self.train_dataloader = Iterator(mc=self.mc, X=X, y=y)

    # Random Seeds (model initialization)
    torch.manual_seed(self.mc.random_seed)
    np.random.seed(self.mc.random_seed)

    # Initialize model
    n_series = self.train_dataloader.n_series
    self.instantiate_estransformer(exogenous_size, n_series)

    # Validating frequencies
    X_train_frequency = pd.infer_freq(X_df.head()['ds'])
    y_train_frequency = pd.infer_freq(y_df.head()['ds'])
    self.frequencies = [X_train_frequency, y_train_frequency]


    if (X_test_df is not None) and (y_test_df is not None):
        X_test_frequency = pd.infer_freq(X_test_df.head()['ds'])
        y_test_frequency = pd.infer_freq(y_test_df.head()['ds'])
        self.frequencies += [X_test_frequency, y_test_frequency]

    assert len(set(self.frequencies)) <= 1, \
      "Match the frequencies of the dataframes {}".format(self.frequencies)

    self.mc.frequency = self.frequencies[0]
    print("Infered frequency: {}".format(self.mc.frequency))

    # Train model
    self._fitted = True
    self.train(dataloader=self.train_dataloader, max_epochs=self.mc.max_epochs,
               warm_start=warm_start, shuffle=shuffle, verbose=verbose)

  def train(self, dataloader, max_epochs, warm_start=False, shuffle=True, verbose=True):

    if self.mc.ensemble: self.estransformer_ensemble = [deepcopy(self.estransformer).to(self.mc.device)] * 5
    if verbose: print(15*'='+' Training ESTransformer  ' + 15*'=' + '\n')

    # Model parameters
    es_parameters = filter(lambda p: p.requires_grad, self.estransformer.es.parameters())
    params = sum([np.prod(p.size()) for p in es_parameters])
    print('Number of parameters of ES: ', params)

    trans_parameters = filter(lambda p: p.requires_grad, self.estransformer.transformer.parameters())
    params = sum([np.prod(p.size()) for p in trans_parameters])
    print('Number of parameters of Transformer: ', params)

    # Optimizers
    if not warm_start:
      self.es_optimizer = optim.Adam(params=self.estransformer.es.parameters(),
                                lr=self.mc.learning_rate*self.mc.per_series_lr_multip,
                                betas=(0.9, 0.999), eps=self.mc.gradient_eps)

      self.es_scheduler = StepLR(optimizer=self.es_optimizer, step_size=self.mc.lr_scheduler_step_size, gamma=0.9)

      self.transformer_optimizer = optim.Adam(params=self.estransformer.transformer.parameters(),
                                              lr=self.mc.learning_rate, betas=(0.9, 0.999), eps=self.mc.gradient_eps,
                                              weight_decay=self.mc.transformer_weight_decay)

      self.transformer_scheduler = StepLR(optimizer=self.transformer_optimizer, 
                                          step_size=self.mc.lr_scheduler_step_size, gamma=self.mc.lr_decay)

    # Loss Functions
    train_tau = self.mc.training_percentile / 100
    train_loss = SmylLoss(tau=train_tau, level_variability_penalty=self.mc.level_variability_penalty)

    eval_tau = self.mc.testing_percentile / 100
    eval_loss = PinballLoss(tau=eval_tau)

    for epoch in range(max_epochs):
      self.estransformer.train()
      start = time.time()
      if shuffle: dataloader.shuffle_dataset(random_seed=epoch)
      losses = []
      for j in range(dataloader.n_batches):
        self.es_optimizer.zero_grad()
        self.transformer_optimizer.zero_grad()

        batch = dataloader.get_batch()
        windows_y, windows_y_hat, levels = self.estransformer(batch)

        # Pinball loss on normalized values
        loss = train_loss(windows_y, windows_y_hat, levels)
        losses.append(loss.data.cpu().numpy())
        loss.backward()
        self.transformer_optimizer.step()
        self.es_optimizer.step()

      # Decay learning rate
      self.es_scheduler.step()
      self.transformer_scheduler.step()

      if self.mc.ensemble:
        copy_estransformer = deepcopy(self.estransformer)
        copy_estransformer.eval()
        self.estransformer_ensemble.pop(0)
        self.estransformer_ensemble.append(copy_estransformer)


      # Evaluation
      self.train_loss = np.mean(losses)
      if verbose:
        print("========= Epoch {} finished =========".format(epoch))
        print("Training time: {}".format(round(time.time()-start, 5)))
        print("Training loss ({} prc): {:.5f}".format(self.mc.training_percentile, self.train_loss))

      if (epoch % self.mc.freq_of_test == 0) and (self.mc.freq_of_test > 0):
        if self.y_test_df is not None:
          self.test_loss = self.model_evaluation(dataloader, eval_loss)
          print("Testing loss  ({} prc): {:.5f}".format(self.mc.testing_percentile, self.test_loss))
          self.evaluate_model_prediction(self.y_train_df, self.X_test_df, self.y_test_df, self.y_hat_benchmark, epoch=epoch)
          self.estransformer.train()

    if verbose: print('Train finished! \n')

  def predict(self, X_df, decomposition=False):
    assert type(X_df) == pd.core.frame.DataFrame
    assert 'unique_id' in X_df
    assert self._fitted, "Model not fitted yet"

    self.estransformer.eval()

    # Create fast dataloader
    if self.mc.n_series < self.mc.batch_size_test: new_batch_size = self.mc.n_series
    else: new_batch_size = self.mc.batch_size_test
    self.train_dataloader.update_batch_size(new_batch_size)
    dataloader = self.train_dataloader

    # Create Y_hat_panel placeholders
    output_size = self.mc.output_size
    n_unique_id = len(dataloader.sort_key['unique_id'])
    panel_unique_id = pd.Series(dataloader.sort_key['unique_id']).repeat(output_size)

    #access column with last train date
    panel_last_ds = pd.Series(dataloader.X[:, 2])
    panel_ds = []
    for i in range(len(panel_last_ds)):
      ranges = pd.date_range(start=panel_last_ds[i], periods=output_size+1, freq=self.mc.frequency)
      panel_ds += list(ranges[1:])

    panel_y_hat= np.zeros((output_size * n_unique_id))

    # Predict
    count = 0
    for j in range(dataloader.n_batches):
      batch = dataloader.get_batch()
      batch_size = batch.y.shape[0]

      if self.mc.ensemble:
        y_hat = torch.zeros((5,batch_size,output_size))
        for i in range(5):
          y_hat[i,:,:] = self.estransformer_ensemble[i].predict(batch)
        y_hat = torch.mean(y_hat,0)
      else:
        y_hat = self.estransformer.predict(batch)

      y_hat = y_hat.data.cpu().numpy()

      panel_y_hat[count:count+output_size*batch_size] = y_hat.flatten()
      count += output_size*batch_size

    Y_hat_panel_dict = {'unique_id': panel_unique_id,
                        'ds': panel_ds,
                        'y_hat': panel_y_hat}

    assert len(panel_ds) == len(panel_y_hat) == len(panel_unique_id)

    Y_hat_panel = pd.DataFrame.from_dict(Y_hat_panel_dict)

    if 'ds' in X_df:
      Y_hat_panel = X_df.merge(Y_hat_panel, on=['unique_id', 'ds'], how='left')
    else:
      Y_hat_panel = X_df.merge(Y_hat_panel, on=['unique_id'], how='left')

    self.train_dataloader.update_batch_size(self.mc.batch_size)
    return Y_hat_panel

  def per_series_evaluation(self, dataloader, criterion):
    with torch.no_grad():
      # Create fast dataloader
      if self.mc.n_series < self.mc.batch_size_test: new_batch_size = self.mc.n_series
      else: new_batch_size = self.mc.batch_size_test
      dataloader.update_batch_size(new_batch_size)

      per_series_losses = []
      for j in range(dataloader.n_batches):
        batch = dataloader.get_batch()
        windows_y, windows_y_hat, _ = self.estransformer(batch)
        loss = criterion(windows_y, windows_y_hat)
        per_series_losses += loss.data.cpu().numpy().tolist()

      dataloader.update_batch_size(self.mc.batch_size)
    return per_series_losses

  def model_evaluation(self, dataloader, criterion):
    with torch.no_grad():
      # Create fast dataloader
      if self.mc.n_series < self.mc.batch_size_test: new_batch_size = self.mc.n_series
      else: new_batch_size = self.mc.batch_size_test
      dataloader.update_batch_size(new_batch_size)

      model_loss = 0.0
      for j in range(dataloader.n_batches):
        batch = dataloader.get_batch()
        windows_y, windows_y_hat, _ = self.estransformer(batch)
        loss = criterion(windows_y, windows_y_hat)
        model_loss += loss.data.cpu().numpy()

      model_loss /= dataloader.n_batches
      dataloader.update_batch_size(self.mc.batch_size)
    return model_loss

  def evaluate_model_prediction(self, y_train_df, X_test_df, y_test_df, y_hat_benchmark='y_hat_naive2', epoch=None):
    assert self._fitted, "Model not fitted yet"

    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_benchmark_panel = y_test_df.filter(['unique_id', 'ds', y_hat_benchmark])
    y_benchmark_panel.rename(columns={y_hat_benchmark: 'y_hat'}, inplace=True)
    y_hat_panel = self.predict(X_test_df)
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel,
                                             y_benchmark_panel, y_insample,
                                             seasonality=self.mc.naive_seasonality)

    if self.min_owa > model_owa:
      self.min_owa = model_owa
      if epoch is not None: self.min_epoch = epoch

    print('OWA: {} '.format(np.round(model_owa, 3)))
    print('SMAPE: {} '.format(np.round(model_smape, 3)))
    print('MASE: {} '.format(np.round(model_mase, 3)))

    return model_owa, model_mase, model_smape

  def long_to_wide(self, X_df, y_df):
    data = X_df.copy()
    data['y'] = y_df['y'].copy()
    sorted_ds = np.sort(data['ds'].unique())
    ds_map = {}
    for dmap, t in enumerate(sorted_ds):
        ds_map[t] = dmap
    data['ds_map'] = data['ds'].map(ds_map)
    data = data.sort_values(by=['ds_map','unique_id'])
    df_wide = data.pivot(index='unique_id', columns='ds_map')['y']

    x_unique = data[['unique_id', 'x']].groupby('unique_id').first()
    last_ds =  data[['unique_id', 'ds']].groupby('unique_id').last()
    assert len(x_unique)==len(data.unique_id.unique())
    df_wide['x'] = x_unique
    df_wide['last_ds'] = last_ds
    df_wide = df_wide.reset_index().rename_axis(None, axis=1)

    ds_cols = data.ds_map.unique().tolist()
    X = df_wide.filter(items=['unique_id', 'x', 'last_ds']).values
    y = df_wide.filter(items=ds_cols).values

    return X, y

  def get_dir_name(self, root_dir=None):
    if not root_dir:
      assert self.mc.root_dir
      root_dir = self.mc.root_dir

    data_dir = self.mc.dataset_name
    model_parent_dir = os.path.join(root_dir, data_dir)
    model_path = ['estransformer_{}'.format(str(self.mc.copy))]
    model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    return model_dir

  def save(self, model_dir=None, copy=None):
    if copy is not None:
        self.mc.copy = copy

    if not model_dir:
      assert self.mc.root_dir
      model_dir = self.get_dir_name()

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    transformer_filepath = os.path.join(model_dir, "transformer.model")
    es_filepath = os.path.join(model_dir, "es.model")

    print('Saving model to:\n {}'.format(model_dir)+'\n')
    torch.save({'model_state_dict': self.es.state_dict()}, es_filepath)
    torch.save({'model_state_dict': self.transformer.state_dict()}, transformer_filepath)

  def load(self, model_dir=None, copy=None):
    if copy is not None:
      self.mc.copy = copy

    if not model_dir:
      assert self.mc.root_dir
      model_dir = self.get_dir_name()

    transformer_filepath = os.path.join(model_dir, "transformer.model")
    es_filepath = os.path.join(model_dir, "es.model")
    path = Path(es_filepath)

    if path.is_file():
      print('Loading model from:\n {}'.format(model_dir)+'\n')

      checkpoint = torch.load(es_filepath, map_location=self.mc.device)
      self.es.load_state_dict(checkpoint['model_state_dict'])
      self.es.to(self.mc.device)

      checkpoint = torch.load(transformer_filepath, map_location=self.mc.device)
      self.transformer.load_state_dict(checkpoint['model_state_dict'])
      self.transformer.to(self.mc.device)
    else:
      print('Model path {} does not exist'.format(path))