from Utils.Comparison import *
from Utils.Data import *
import os, torch, argparse
from DeployedESTransformer import DeployedESTransformer
from utils_configs import get_config
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tsa.stattools as stattools

dataset_name = ['Daily','Monthly']
max_epoch = [499, 499]

for i in range(len(dataset_name)):
  path = 'D:\\Sang\\hybcast\\hybcast3\\' + dataset_name[i]

  config = get_config(dataset_name[i])

  X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=dataset_name[i], directory='D:\\Sang\\ESRNN_result\\',
                                                                  num_obs=100000)
  if dataset_name[i] == 'Daily': N = 3
  else: N = 1
  #Instantiate the model
  model = DeployedESTransformer(max_epochs = config['train_parameters']['max_epochs'],
                                batch_size = config['train_parameters']['batch_size'],
                                freq_of_test = config['train_parameters']['freq_of_test'],
                                learning_rate = float(config['train_parameters']['learning_rate']),

                                per_series_lr_multip = config['train_parameters']['per_series_lr_multip'],
                                lr_scheduler_step_size = config['train_parameters']['lr_scheduler_step_size'],
                                lr_decay = config['train_parameters']['lr_decay'],

                                transformer_weight_decay = 0.0,

                                level_variability_penalty = config['train_parameters']['level_variability_penalty'],
                                
                                testing_percentile = config['train_parameters']['testing_percentile'],
                                training_percentile = config['train_parameters']['training_percentile'],
                                
                                ensemble = config['train_parameters']['ensemble'], ####################
                                seasonality = config['data_parameters']['seasonality'],
                                
                                random_seed = config['model_parameters']['random_seed'],
                                device= config['device'],
                                root_dir= '/content/gdrive/My Drive/ESTransformer',

                                # Semi-hyperparameters
                                input_size = config['data_parameters']['input_size'],
                                d_input = config['data_parameters']['d_input'], # input_size + 6 (which is exo features)
                                output_size = config['data_parameters']['output_size'], # must be the same as d_output
                                d_output = config['data_parameters']['d_output'], # input_size + output_size >= 13
                                
                                # Worth tuning hyperparameters
                                d_model = 128,
                                q = 4,
                                v = 4,
                                h = 4,
                                N = N,
                                attention_size = 4,
                                dropout = 0,
                                chunk_mode = None, ####################
                                pe = None, ####################
                                pe_period = 24, dataset_name = dataset_name[i]) ####################

  model.preprocess(X_train_df, y_train_df, X_test_df, y_test_df)
  model.load(path + '\\model\\model_epoch_' + str(max_epoch[i]) + '_' + dataset_name[i])

  y_hat_df = model.predict(X_test_df)

  seasonality = config['data_parameters']['seasonality']

  if not seasonality:
      seasonality = 1
  else:
      seasonality = seasonality[0]

  final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df,
                                                              X_test_df, y_test_df,
                                                              naive2_seasonality=seasonality)

  # encoder = model.estransformer.transformer.layers_encoding[0]
  # attn_map = encoder.attention_map[0].cpu()

  # Plot
  # plt.figure(figsize=(20, 20))
  # sns.heatmap(attn_map.detach().numpy())
  # plt.savefig(path+'\\figure\\'+ dataset_name[i] + "_attention_map.png", dpi=500, bbox_inches='tight')
  # plt.show()