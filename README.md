# Sementic Segmentation on  Weizmann Horse dataset
This is the course project of Visual Cognition.

## Data Preparation
1.  Please download Weizmann Horse dataset first from
https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata
1. Put all the horse images in `./data/horse`, and all the mask images in `./data/mask`.

The final structure of datasets should be as following:
```
├─PROJECT_DIR
│ ├─data   
│ │ ├─horse
│ │ │ ├─xxx.png
│ │ │ ├─ ......
│ │ ├─mask
│ │ │ ├─xxx.png
│ │ │ ├─ ......
```
## train 
```python
python main.py \
    --mode train \
    --img_size 224 \
    --output_dir ${PATH_TO_CHECKPOINT} \  
    --save_freq 30 \
    --lr 1.5e-2 \
    --epoch 100 \
    --batch_size 16 
```
Use `--mode train` to train your model.
## test
```python
python main.py \
    --mode test \
    --checkpoint_path ${PATH_TO_SAVED_CHECKPOINT}
```
Use `--mode test` to test your model.
## results
### metrics
| model | mIoU | Boundary IoU |
| :-----:| :----: | :----: |
| UNet++ | 0.892 | 0.530 |

### visualization
The visualization should be like this:
![](https://raw.githubusercontent.com/gitcsq/segmentation-WeizmannHorse/main/sample.png)

