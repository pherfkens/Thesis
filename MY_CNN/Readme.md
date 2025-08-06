## Data Preparation  
Before being able to get results, the right pickles have to be in the folders. So in /application/Dataset/Testing_set/ the files are currently dummy files. Replace those with the one in the zipfile.

## Get testing results
Run /application/Inference.py to get the test results of the final model.

For example, running the Inference.py with the final model "final_model_AD_CNN_dense_layer_hop_combined_monitor_pleasant.pth", would give these results:

```python
Number of 3576 audios in testing
AEC     AUC:  0.83
PAQ_8D_AQ       MSE MEAN: 1.114
pleasant_mse: 1.006 eventful_mse: 1.135 chaotic_mse: 1.107 vibrant_mse: 1.088
uneventful_mse: 1.159 calm_mse: 1.055 annoying_mse: 1.166 monotonous_mse: 1.198 
```
