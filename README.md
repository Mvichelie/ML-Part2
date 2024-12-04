I have two parts for this assignemnt. Part 1 Image classification (Bonus A -class imbalance) and Part 2  autoencoding with clustering (Bonus B
with Visualization of images and coparing it with the original. . 

Run the script to train the classification model with Bonus A:
python trainpart1.py
This script uses a "WeightedRandomSampler" to handle class imbalance (Bonus A)
Trains the model for 20 epochs
The output trained model is saved as wikiart.pth (note that the file is too big to be uploaded here)

To test the classification model;
Run the script below:
python testtrainpart1.py
This script evaluates the model's performance
The output gives you a confusion matrix saved as a png, along with accuracy, precision, recall, F1-score
I included (accuracy, precision, recall and F1 due to the imbalanced classes and high accuracy alone might not
reflect true performance. 

Evaluation and Results:
During training, the loss decreased from ~1102.24 (epoch 1) to ~5.28 (epoch 20), which shows the
that the model was learning effectively.
Accuracy: (~20.79%): slight better but model's ability to generalize is limited
Precision: (~14.26%): model is overpredicting 
Recall: (~13.91%): model fails to identify the correct class for most samples
F1-Score (~13.13%): model stuggles equally with false negatives and false positives
Confusion metrics: shows that some classes performed well but others struggled. The model seemed to predict the majority 
classes more often and sparse predictions for the other classes. 
For improvement, finetuning learning rate, epochs, using random search, upsampling the data. 




Part 2:
In wikiart.py, I implemented an autoencoder which is used for the scripts below. 
The autoencoder was trained to compress images into a latent representation and to reconstruct them as closely as possible. 

To train the autoencoder run the script below:
python trainauto.py
This script trains an autoencoder for 20 epochs to minimize image reconstruction loss
The model is saved as autoencoder.pth (model not here due to file size)
Training:
The mean squared error between the original and reconstructed images was minimized. This was used as the 
loss function because it evaluates how closely the reconstructed image matches the original (pixel wise)
It ensures that the autoencoder focuses on learning fine details. 

To test (Bonus B)
run to the below:
python testtrainauto.py
This script extract latent encodings and perfroms PCA for dimensionality reduction
It also uses Agglomerative clustering which gives the output of latent space clustering (clusters.png)
And a reconstructed image comparison of the original and reconstructed side-by-side called.(reconstruction.ong) 

Results:
Training:
The loss decreased from ~104.59 (epoch 1) to ~84.49 (epoch 20), which shows that the autoencoder was learning effectively to reconstruct the 
images while preserving key features.

Clustering results:
The autoencoder's latent representations were clustered into 27 groups, corresponding to the number of art styles in the dataset. These representations were reduced to 2D using PCA for visualization, 
producing the scatter plot (clusters.png).
![image](https://github.com/user-attachments/assets/5d28e288-7f5e-4a43-9324-3ee035ad0a54)
The clusters show several distinct groupings, which suggests that the autoencoder captured style-specific features from the dataset.
There is some overlap between the clusters, which I assume isfrom the shared characteristics among certain art styles. 

Reconstruction results:
The image(s) show that the autoencoder has the ability to preserve the overall structure of the input image
![image](https://github.com/user-attachments/assets/15dd13e4-8a9c-4612-9425-134d931d5c06)

Overall the autoencoder successfully compresses the images while preserving the major features, The clustering
results show that the autoencoder has the ability to differentiate between art styles. 









