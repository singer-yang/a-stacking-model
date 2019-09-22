# wine-quality
 ## a stacking model for wine quality prediction
 
![image](https://github.com/singer-yang/wine-quality/blob/master/model.jpg)

## model description

The only difference between M6 and M1-M5 is, that M6 is trained on the entire original training data, whereas M1-M5 are trained only on 4 out of 5 folds.

With M1-M5 you can build valid out-of-fold predictions for the training set (the orange ones) to form a "new feature" for the 2nd layer (not possible with M6). You can also predict the test set with M1-M5 to get 5 sets of test set predictions .. but you only need one set of test set predictions for the corresponding feature to the orange out-of-fold train set predictions.

Hence, you reduce those 5 sets to 1 by averaging. That's the first variant. Alternatively, you train M6 and use its test set predictions as feature for the 2nd layer (instead of the average of the test set predictions from M1-M5).
