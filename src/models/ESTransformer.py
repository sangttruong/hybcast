import torch.nn as nn

import sys, os
sys.path.append(os.getcwd()+"\\src\\models\\")

# Submodel
from Transformer import Transformer
from ES import ES

class ESTransformer(nn.Module):
  def __init__(self, mc):
    super().__init__()
    self.mc = mc
    self.es = ES(mc).to(self.mc.device)
    self.transformer = Transformer(mc).to(self.mc.device)

  def forward(self, ts_object):
    windows_y_hat, windows_y, levels, seasonalities = self.es(ts_object)
    windows_y_hat = self.transformer(windows_y_hat)
    return windows_y, windows_y_hat, levels

  def predict(self, ts_object):
    windows_y_hat, _, levels, seasonalities = self.es(ts_object)
    windows_y_hat = self.transformer(windows_y_hat)
    trend = windows_y_hat[-1,:,:] # Last observation prediction
    y_hat = self.es.predict(trend, levels, seasonalities)
    return y_hat
