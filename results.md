#### Fine-tune
|model|pre-trained|top-1|top-3|setting|
|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet50|places365||94.2|--depth 32|
|ResNet152|places365||94.4||
|ResNet50|places365|80.18|94.48|--lr 0.01 --lr-decay 10 --weight-decay 0 --moment 0 |
|ResNet50|places365|81.32|94.85|--lr 0.05 --lr-decay 10 --weight-decay 0 --moment 0 |
|ResNet50|places365|81.00|95.00|--lr 0.02 --lr-decay 10 --weight-decay 0 --moment 0 |
|ResNet50|places365|81.63|95.15|--lr 0.03 --lr-decay 10 --weight-decay 0 --moment 0 |