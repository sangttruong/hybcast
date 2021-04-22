from DeployedESTransformer import DeployedESTransformer
from Utils.Comparison import *
from Utils.Data import *
from utils_configs import get_config

import os, torch, argparse

def main(args):
  config = get_config(args.dataset)
  if config['data_parameters']['frequency'] == 'Y':
    config['data_parameters']['frequency'] = None

  #Setting needed parameters
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

  if args.num_obs: num_obs = args.num_obs
  else: num_obs = 100000

  if args.use_cpu == 1: config['device'] = 'cpu'
  else: assert torch.cuda.is_available(), 'No cuda devices detected. You can try using CPU instead.'
  
  print('Reading data')
  X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=args.dataset,
                                                                 directory=args.results_directory,
                                                                 num_obs=num_obs)
                                                       
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
                                root_dir= args.results_directory,

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
                                N = 2,
                                attention_size = 4,
                                dropout = 0,
                                chunk_mode = None, ####################
                                pe = None, ####################
                                pe_period = 24) ####################
  # Fit model
  # If y_test_df is provided the model # will evaluate predictions on this set every freq_test epochs
  model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

  # Predict on test set
  y_hat_df = model.predict(X_test_df)

  # Evaluate predictions
  final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df, X_test_df, y_test_df, naive2_seasonality=1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Replicate M4 results for the ESRNN model')
  parser.add_argument("--dataset", required=True, type=str,
                      choices=['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Hourly', 'Daily'],
                      help="set of M4 time series to be tested")
  parser.add_argument("--results_directory", required=True, type=str,
                      help="directory where M4 data will be downloaded")
  parser.add_argument("--gpu_id", required=False, type=int,
                      help="an integer that specify which GPU will be used")
  parser.add_argument("--use_cpu", required=False, type=int,
                      help="1 to use CPU instead of GPU (uses GPU by default)")
  parser.add_argument("--num_obs", required=False, type=int,
                      help="number of M4 time series to be tested (uses all data by default)")
  parser.add_argument("--test", required=False, type=int,
                      help="run fast for tests (no test by default)")
  args = parser.parse_args()

  main(args)
  
  