import torch
import torch.nn as nn
import numpy as np


class ES(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.mc = mc

        # Level and Seasonality Smoothing parameters
        # 1 level, S seasonalities, S init_seas
        embeds_size = 1 + len(self.mc.seasonality) + sum(self.mc.seasonality)
        init_embeds = torch.ones((self.mc.n_series, embeds_size)) * 0.5
        self.embeds = nn.Embedding(self.mc.n_series, embeds_size)
        self.embeds.weight.data.copy_(init_embeds)
        self.register_buffer(
            'seasonality', torch.LongTensor(self.mc.seasonality))
        assert len(self.mc.seasonality) in [0, 1, 2]

    # Forward function
    def forward(self, ts_object):
        y = ts_object.y
        idxs = ts_object.idxs
        n_series, n_time = y.shape
        batch_size = len(idxs)

        if self.training:
            windows_end = n_time-self.mc.input_size-self.mc.output_size+1
            windows_range = range(windows_end)
        else:
            windows_start = n_time-self.mc.input_size-self.mc.output_size+1
            windows_end = n_time-self.mc.input_size+1
            windows_range = range(windows_start, windows_end)

        n_windows = len(windows_range)
        assert n_windows > 0

        # Initialize windows, levels and seasonalities
        levels, seasonalities = self.compute_levels_seasons(y, idxs)
        windows_y_hat = torch.zeros(
            (n_windows, batch_size, self.mc.input_size+self.mc.exogenous_size), device=self.mc.device)
        windows_y = torch.zeros(
            (n_windows, batch_size, self.mc.output_size), device=self.mc.device)

        for i, window in enumerate(windows_range):
            # Windows yhat
            y_hat_start = window
            y_hat_end = self.mc.input_size + window

            # Y_hat deseasonalization and normalization
            window_y_hat = self.normalize(
                y=y[:, y_hat_start:y_hat_end],
                level=levels[:, [y_hat_end-1]],
                seasonalities=seasonalities,
                start=y_hat_start,
                end=y_hat_end
            )

            if self.training:
                window_y_hat = self.gaussian_noise(
                    window_y_hat, std=self.mc.noise_std
                )

            # Concatenate categories
            if self.mc.exogenous_size > 0:
                window_y_hat = torch.cat(
                    (window_y_hat, ts_object.categories), 1
                )

            windows_y_hat[i, :, :] += window_y_hat

            # Windows y (for loss during train)
            if self.training:
                y_start = y_hat_end
                y_end = y_start+self.mc.output_size
                # Y deseasonalization and normalization
                window_y = self.normalize(
                    y=y[:, y_start:y_end],
                    level=levels[:, [y_start]],
                    seasonalities=seasonalities,
                    start=y_start, end=y_end
                )
                windows_y[i, :, :] += window_y

        return windows_y_hat, windows_y, levels, seasonalities

    # Compute Gaussian noise
    def gaussian_noise(self, input_data, std=0.2):
        size = input_data.size()
        noise = torch.autograd.Variable(
            input_data.data.new(size).normal_(0, std))
        return input_data + noise

    # Compute levels and seasonality
    def compute_levels_seasons(self, y, idxs):
        # Lookup parameters per serie
        embeds = self.embeds(idxs)
        lev_sms = torch.sigmoid(embeds[:, 0])

        # Initialize seasonalities
        seas_prod = torch.ones(len(y[:, 0])).to(y.device)
        seasonalities1 = []
        seasonalities2 = []
        seas_sms1 = torch.ones(1).to(y.device)
        seas_sms2 = torch.ones(1).to(y.device)

        if len(self.seasonality) > 0:
            seas_sms1 = torch.sigmoid(embeds[:, 1])
            init_seas1 = torch.exp(
                embeds[:, 2:(2+self.seasonality[0])]).unbind(1)
            assert len(init_seas1) == self.seasonality[0]

            for i in range(len(init_seas1)):
                seasonalities1 += [init_seas1[i]]
            seasonalities1 += [init_seas1[0]]
            seas_prod = seas_prod * init_seas1[0]

        if len(self.seasonality) == 2:
            seas_sms2 = torch.sigmoid(embeds[:, 2+self.seasonality[0]])
            init_seas2 = torch.exp(embeds[:, 3+self.seasonality[0]:]).unbind(1)
            assert len(init_seas2) == self.seasonality[1]

            for i in range(len(init_seas2)):
                seasonalities2 += [init_seas2[i]]
            seasonalities2 += [init_seas2[0]]
            seas_prod = seas_prod * init_seas2[0]

        # Initialize levels
        levels = []
        levels += [y[:, 0]/seas_prod]

        # Recursive seasonalities and levels
        ys = y.unbind(1)
        n_time = len(ys)
        for t in range(1, n_time):
            seas_prod_t = torch.ones(len(y[:, t])).to(y.device)
            if len(self.seasonality) > 0:
                seas_prod_t = seas_prod_t * seasonalities1[t]
            if len(self.seasonality) == 2:
                seas_prod_t = seas_prod_t * seasonalities2[t]

            newlev = lev_sms * (ys[t] / seas_prod_t) + \
                (1-lev_sms) * levels[t-1]
            levels += [newlev]

            if len(self.seasonality) == 1:
                newseason1 = seas_sms1 * \
                    (ys[t] / newlev) + (1-seas_sms1) * seasonalities1[t]
                seasonalities1 += [newseason1]

            if len(self.seasonality) == 2:
                newseason1 = seas_sms1 * (ys[t] / (newlev * seasonalities2[t])) + \
                                         (1-seas_sms1) * seasonalities1[t]
                seasonalities1 += [newseason1]
                newseason2 = seas_sms2 * (ys[t] / (newlev * seasonalities1[t])) + \
                                         (1-seas_sms2) * seasonalities2[t]
                seasonalities2 += [newseason2]

        levels = torch.stack(levels).transpose(1, 0)
        seasonalities = []

        if len(self.seasonality) > 0:
            seasonalities += [torch.stack(seasonalities1).transpose(1, 0)]
        if len(self.seasonality) == 2:
            seasonalities += [torch.stack(seasonalities2).transpose(1, 0)]

        return levels, seasonalities

    # Deseasonlization and normalization
    def normalize(self, y, level, seasonalities, start, end):
        y_n = y / level
        for s in range(len(self.seasonality)):
            y_n /= seasonalities[s][:, start:end]
        y_n = torch.log(y_n)
        return y_n

    # Predict value based on trend, levels, seasonalities
    def predict(self, trend, levels, seasonalities):
        seasonality = self.mc.seasonality
        n_time = levels.shape[1]

        # Denormalize
        trend = torch.exp(trend)

        # Completion of seasonalities if prediction horizon is larger than seasonality
        # Naive2 like prediction, to avoid recursive forecasting
        for s in range(len(seasonality)):
            if self.mc.output_size > seasonality[s]:
                repetitions = int(
                    np.ceil(self.mc.output_size/seasonality[s]))-1
                last_season = seasonalities[s][:, -seasonality[s]:]
                extra_seasonality = last_season.repeat((1, repetitions))
                seasonalities[s] = torch.cat(
                    (seasonalities[s], extra_seasonality), 1)

        # Deseasonalization and normalization (inverse)
        y_hat = trend * levels[:, [n_time-1]]
        for s in range(len(seasonality)):
            y_hat *= seasonalities[s][:, n_time:(n_time+self.mc.output_size)]

        return y_hat
