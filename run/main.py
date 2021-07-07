import sys, os, inspect, torch, argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from hybcast.models.DeployedESTransformer import DeployedESTransformer
from features.Comparison import *
from data.make_dataset import *
from config import cfg
import argparse


def main(cfg):
  # Checking data directory
  input_directory = cfg['dataset']['input_directory']
  if input_directory == None: 
    data_dir = parentdir.replace("\hybcast", "")
    data_dir = parentdir.replace("/hybcast", "")

    input_directory = data_dir + "\\hybcast\\data\\external"
    if os.path.isdir(input_directory) == False:
      print("Creating data directory " + input_directory)
      os.makedirs(input_directory)
  else: 
    if os.path.isdir(input_directory) == False:
      print("Creating data directory " + input_directory)
      os.makdirs(input_directory)

  # Checking cuda availability
  if torch.cuda.is_available() == False: 
    print("CUDA is not available, using cpu instead")
    cfg['device'] = 'cpu'
  else: print("Using CUDA")


  print('Preparing M4 data - ' + cfg['dataset']['name'])
  X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=cfg['dataset']['name'],
                                                                 directory=input_directory,
                                                                 num_obs=cfg["dataset"]["num_obs"] ) 

  print("Successfully get the M4 data.\nBegin ESTransformer")                                     
  #Instantiate the model
  model = DeployedESTransformer(# Device and dataset
                                device= cfg['device'],
                                root_dir= input_directory,
                                dataset_name = cfg['dataset']['name'],

                                # Train parameters
                                max_epochs = 500,
                                batch_size = cfg['train_parameters']['batch_size'],
                                freq_of_test = cfg['train_parameters']['freq_of_test'],
                                learning_rate = float(cfg['train_parameters']['learning_rate']),
                                per_series_lr_multip = cfg['train_parameters']['per_series_lr_multip'],
                                lr_scheduler_step_size = cfg['train_parameters']['lr_scheduler_step_size'],
                                lr_decay = cfg['train_parameters']['lr_decay'],
                                level_variability_penalty = cfg['train_parameters']['level_variability_penalty'],
                                testing_percentile = cfg['train_parameters']['testing_percentile'],
                                training_percentile = cfg['train_parameters']['training_percentile'],
                                ensemble = cfg['train_parameters']['ensemble'],

                                # ES parameters 
                                seasonality = cfg['ES_parameters']['seasonality'],
                                random_seed = cfg['ES_parameters']['random_seed'],
                                input_size = cfg['ES_parameters']['input_size'],
                                d_input = cfg['ES_parameters']['d_input'], # input_size + 6 (which is exo features)
                                output_size = cfg['ES_parameters']['output_size'], # must be the same as d_output
                                d_output = cfg['ES_parameters']['d_output'], # input_size + output_size >= 13
                                
                                # Transformer parameter
                                transformer_weight_decay = 0.0,
                                d_model = cfg['Transformer_parameter']['d_model'],
                                q = cfg['Transformer_parameter']['q'],
                                v = cfg['Transformer_parameter']['v'],
                                h = cfg['Transformer_parameter']['h'],
                                N = cfg['Transformer_parameter']['N'],
                                attention_size = cfg['Transformer_parameter']['attention_size'],
                                dropout = cfg['Transformer_parameter']['dropout'],
                                chunk_mode = cfg['Transformer_parameter']['chunk_mode'], 
                                pe = cfg['Transformer_parameter']['pe'], 
                                pe_period = cfg['Transformer_parameter']['pe_period']
                                )
  # Fit model
  # If y_test_df is provided the model # will evaluate predictions on this set every freq_test epochs
  model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

  # Predict on test set
  y_hat_df = model.predict(X_test_df)

  # Predict on train set
  y_train_hat_df = model.predict(X_train_df)

  # Evaluate predictions
  final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df, X_test_df, y_test_df, naive2_seasonality=1)

if __name__ == '__main__':
  # Load cmd line args 
  """Parses the arguments."""
  parser = argparse.ArgumentParser(description='Replicate M4 results for the ESRNN model')

  parser.add_argument(
    '--cfg',
    dest='cfg_file',
    help='cfg file path',
    required=True,
    type=str
  )

  args = parser.parse_args()
  cfg.merge_from_file(args.cfg_file)
  main(cfg)
  
  
  