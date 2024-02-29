# Computer Vision Methods to Predict Stock 
### Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date
3. Use window 100, i.e. predict with last 100 dates’ data, as X value
4. Expand X shape (5, 100) to (5, 100, 100) and cosine values
5. Predict the next dates’ Open change and Close change (percentile)

### Models
- Convolution Model
    - ResNet
    - GoogleNet
    - DenseNet
- Transformer Based
    - VisionTransformer + CNN (VTCNN)
    - Conformer + CNN (ConCNN)
    - Conformer + ResNet (ConRes)
- Pretrain Models (Finetune)
    > NOTE: Pretrained requires input size 224x224, ..., different from 100x100, reprocess data in train file
    - VisionTransformer: ViT_b_16

### Experiments
Buy if (predicted next day’s Close - Open > today’s Close * 0.004)
|  | ResNet | GoogleNet | DenseNet | VTCNN | ConCNN | ConRes |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy |  |  |  |  |  |  |
| Final Asset |  |  |  |  |  |  |

|             | ViT_b_16 |     |     |         |        |       |
| ----------- | -------- | --- | --- | --- | --- | --- |
| Accuracy    |          |      |     |     |  |  |
| Final Asset |          |  |  |  |  |  |

### Plots
<details>
<summary>ResNet</summary>
</details>
<details>
<summary>GoogleNet</summary>
</details>
<details>
<summary>DenseNet</summary>
</details>
<details>
<summary>VTCNN</summary>
</details>
<details>
<summary>ConCNN</summary>
</details>
<details>
<summary>ConRes</summary>
</details>