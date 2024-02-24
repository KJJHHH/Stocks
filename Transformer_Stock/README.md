# Computer Vision Methods to Predict Stock 
### Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date as do, dc, ...
3. Normalise with train set
4. Select do, dc, dh, dl, dvm and Close
5. Use window 400, i.e. predict with last 400 dates’ data, as X value
6. Predict the next dates’ (Open - Close)/Close

### Models
- Transformer Based
    - Transformer Encoder
    - Transoformer EncodDecoder
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