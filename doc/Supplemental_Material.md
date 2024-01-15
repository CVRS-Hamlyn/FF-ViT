# Supplemental Material
The convergence and stability study & description of blur metrics are avaliable at [here](https://https://github.com/CVRS-Hamlyn/FF-ViT/blob/main/doc/Supplemental_material.pdf).

## SOTA Comparison

| Model| | Lens Paper| | | Cow Heart| |
|---|---|---|---|---|---|---|
|| MAE | $\sigma$ | $Acc_{dir}$ |  MAE | $\sigma$ | $Acc_{dir}$ |
ResNet 18|1.142|0.485|97.5%|2.393|1.166|91.9%|
ConvNeXt|1.486|0.628|94.7%|2.722|1.404|84.1%|
SFFC-Net|1.164|0.659|97.4%|2.227|0.771|95.6%|
Swin-T|1.143|0.473|96.2%|2.795|0.781|90.5%|
Swin-S|1.174|0.552|95.8%|3.573|1.014|88.2%|
Swin-B|1.227|0.505|94.9%|3.681|1.284|84.9%|
XCiT-T|1.091|0.590|95.8%|2.121|0.961|89.1%|
XCiT-S|**1.05**|**0.468**|96.7%|2.496|0.933|91.6%|
XCiT-M|1.260|0.490|95.9%|3.304|1.731|82.5%|
FF-ViT(S)|1.184|0.539|93.8%|1.795|0.659|93.1%|
FF-ViT(B)|1.143|0.521|96.1%|1.876|0.524|92.9%|
FF-ViT(T)|1.167|0.469|**98.5%**|**1.549**|**0.336**|**97.0%**|
