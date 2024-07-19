## Simple  
- 70000 steps後最佳accuracy為0.7552  
## Medium  
- 100000 steps後最佳accuracy為0.8844  
- 修改方向大多與作業說明中相反
  1. d_model從40增加至128  
  2. 加入dropout並設為0.1  
  3. 使用三層TransformerEncoderLayer  
  4. pred_layer中加入Batch Normalization  
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
## Hard  
- Conformer  
  ![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*8vgwJxJDmu0cR7aL9pYWJQ.png)
  1. Feed Forward Module
  ```python
  self.FeedForwardLayer1 = nn.Sequential(
    nn.Linear(d_model, 2048),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(2048, d_model)
  )
  ```
  1. Multi-Head Self Attention Module
  ```python
  self.linear_k = nn.Linear(d_model, d_model, bias=False)
  self.linear_q = nn.Linear(d_model, d_model, bias=False)
  self.linear_v = nn.Linear(d_model, d_model, bias=False)
  ```
  1. Convolution Module
  ```python
  self.pointwiseConv1 = nn.Conv1d(d_model, d_model, kernel_size=(1,))
  self.depthwiseConv = nn.Conv1d(1, 1, kernel_size=(31,), padding=(15,))
  self.pointwiseConv2 = nn.Conv1d(d_model, d_model, kernel_size=(1,))
  ```
  1. Feed Forward Module
  ```python
      self.FeedForwardLayer2 = nn.Sequential(
        nn.Linear(d_model, 2048),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(2048, d_model),
        nn.LayerNorm(d_model)
      )
  ```
  1. Layernorm
  ```python
  self.Acti_Norm = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.ReLU()
  )
  ```

- 將TransformerEncoderLayer與TransformerEncoder改為一個ConformerBlock，設定如下方程式
- 100000 steps後最佳accuracy為0.8954  
  (dim_head = heads = 5 accuracy=0.8851)
```python
from conformer import ConformerBlock
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

