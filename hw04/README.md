## Simple  

70000 steps後最佳accuracy為0.7552
## Medium  
```python
class Classifier(nn.Module):
  def __init__(self, d_model=128, n_spks=600, dropout=0.1):
    super().__init__()
    self.prenet = nn.Linear(40, d_model)
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=128, nhead=2, dropout=dropout
    )
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.BatchNorm1d(d_model),
      nn.ReLU(),
      nn.Linear(d_model, n_spks),
    )

  def forward(self, mels):
    out = self.prenet(mels)
    out = out.permute(1, 0, 2)
    out = self.encoder(out)
    out = out.transpose(0, 1)
    stats = out.mean(dim=1)
    out = self.pred_layer(stats)
    return out
```
100000 steps後最佳accuracy為0.8844
## Hard  
```python
class Classifier(nn.Module):
  def __init__(self, d_model=256, n_spks=600, dropout=0.2):
    super().__init__()
    self.prenet = nn.Linear(40, d_model)
    self.encoder = ConformerBlock(
			dim = d_model,
			dim_head = 4,
			heads = 4,
      ff_mult = 2,
			attn_dropout = dropout,
			ff_dropout = dropout,
			conv_dropout = dropout,
		)
    self.pred_layer = nn.Sequential(
      nn.BatchNorm1d(d_model),
      nn.ReLU(),
      nn.Linear(d_model, n_spks)
    )

  def forward(self, mels):
    out = self.prenet(mels)
    out = out.permute(1, 0, 2)
    out = self.encoder(out)
    out = out.transpose(0, 1)
    stats = out.mean(dim=1)
    out = self.pred_layer(stats)
    return out
```
100000 steps後最佳accuracy為0.8954
(dim_head = heads = 5 accuracy=0.8851)
