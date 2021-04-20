from EsTransformer import *
from Utils.Comparison import *
from Utils.Data import *
from Utils.Parsing import parse_train_args


# path = 'D:\\Sang\\hybcast\\hybcast3\\data'
args = parse_train_args()

for dataset_name in ['Yearly', 'Quarterly', 'Monthly', 'Hourly', 'Weekly', 'Daily']:
  X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name=dataset_name, directory = args.data_path, num_obs=1000000)
  
  input = 6
  
  model = ESTransformer(max_epochs=32,
                        batch_size = 32,
                        freq_of_test=1,
                        learning_rate=1e-3,

                        per_series_lr_multip=0.8,
                        lr_scheduler_step_size=10,
                        lr_decay=0.1,
                        transformer_weight_decay=0.0,

                        level_variability_penalty=100,
                        
                        testing_percentile=50,
                        training_percentile=50,
                        
                        ensemble=False, ####################
                        seasonality=[],
                        
                        random_seed=1,
                        device='cuda',
                        root_dir= args.data_path,

                        # Semi-hyperparameters
                        input_size = input,
                        d_input = input + 6, # input_size + 6 (which is exo features)
                        output_size = 13-input, # must be the same as d_output
                        d_output = 13-input, # input_size + output_size >= 13
                        
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