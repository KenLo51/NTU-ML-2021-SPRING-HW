## Easy  
`DataLoader`中的`num_workers`從8改為0  
其餘不做任何修改200-epoch後accuracy達到0.503  
```python
# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```
## Medium  
將訓練資料經由以下轉換做資料擴增後再訓練模型  
僅加入資料擴增後並在200-epoch後accuracy達到0.572  
```python
augmentationTransforms = transforms.Compose([
    transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter()
])
```
## Hard  
加深Classifier使用的模型，與模型在每個epoch紀錄一個checkpoint、loss、accuracy  
在第302-epoch結束後accuracy可達0.755952418  
```python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # image size: [64, 64, 64]
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # image size: [128, 32, 32]
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # image size: [256, 16, 16]
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # image size: [512, 8, 8]
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # image size: [1024, 4, 4]
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # image size: [1024, 2, 2]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024*2*2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(), 
            nn.Dropout(0.4),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x
```
