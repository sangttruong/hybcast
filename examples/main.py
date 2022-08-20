import torch
from argparse import Namespace, ArgumentParser
from src.config import cfg
from src.make_dataset import prepare_m4_data
from src.comparison import evaluate_prediction_owa
from src.models.DeployedESTransformer import DeployedESTransformer

if __name__ == '__main__':

    # arguments
    parser = ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg_file)
    if not torch.cuda.is_available():
        cfg['device'] = 'cpu'

    config = Namespace(
        # Device and dataset
        device=cfg['device'],
        root_dir=cfg['dataset']['input_directory'],
        dataset_name=cfg['dataset']['name'],

        # Train parameters
        max_epochs=cfg['train_parameters']['max_epochs'],
        batch_size=cfg['train_parameters']['batch_size'],
        freq_of_test=cfg['train_parameters']['freq_of_test'],
        learning_rate=float(cfg['train_parameters']['learning_rate']),
        per_series_lr_multip=cfg['train_parameters']['per_series_lr_multip'],
        lr_scheduler_step_size=cfg['train_parameters']['lr_scheduler_step_size'],
        lr_decay=cfg['train_parameters']['lr_decay'],
        level_variability_penalty=cfg['train_parameters']['level_variability_penalty'],
        testing_percentile=cfg['train_parameters']['testing_percentile'],
        training_percentile=cfg['train_parameters']['training_percentile'],
        ensemble=cfg['train_parameters']['ensemble'],

        # ES parameters
        seasonality=cfg['ES_parameters']['seasonality'],
        random_seed=cfg['ES_parameters']['random_seed'],
        input_size=cfg['ES_parameters']['input_size'],
        # input_size + 6 (which is exo features)
        d_input=cfg['ES_parameters']['d_input'],
        # must be the same as d_output
        output_size=cfg['ES_parameters']['output_size'],
        # input_size + output_size >= 13
        d_output=cfg['ES_parameters']['d_output'],

        # Transformer parameter
        transformer_weight_decay=cfg['Transformer_parameter']['transformer_weight_decay'],
        d_model=cfg['Transformer_parameter']['d_model'],
        q=cfg['Transformer_parameter']['q'],
        v=cfg['Transformer_parameter']['v'],
        h=cfg['Transformer_parameter']['h'],
        N=cfg['Transformer_parameter']['N'],
        attention_size=cfg['Transformer_parameter']['attention_size'],
        dropout=cfg['Transformer_parameter']['dropout'],
        chunk_mode=cfg['Transformer_parameter']['chunk_mode'],
        pe=cfg['Transformer_parameter']['pe'],
        pe_period=cfg['Transformer_parameter']['pe_period']
    )

    config.batch_size_test = 64
    config.gradient_eps = 1e-8
    config.noise_std = 0.001
    config.frequency = None
    config.max_periods = 20
    config.input_size_i = config.input_size
    config.output_size_i = config.output_size
    config.min_series_length = config.input_size_i + config.output_size_i
    config.max_series_length = (
        config.max_periods * config.input_size) + config.min_series_length

    if len(config.seasonality) > 0:
        config.naive_seasonality = config.seasonality[0]
    else:
        config.naive_seasonality = 1

    # data
    X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(
        dataset_name=config.dataset_name,
        directory=cfg['dataset']['input_directory'],
        num_obs=cfg["dataset"]["num_obs"]
    )

    # model
    model = DeployedESTransformer(config)

    # Train
    model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

    # Evaluatation
    y_train_hat_df = model.predict(X_train_df)
    y_hat_df = model.predict(X_test_df)
    final_owa, final_mase, final_smape = evaluate_prediction_owa(
        y_hat_df,
        y_train_df,
        X_test_df,
        y_test_df,
        naive2_seasonality=1
    )
