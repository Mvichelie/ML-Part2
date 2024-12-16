I have two parts for this assignemnt. Part 1 Image classification (Bonus A -class imbalance) and Part 2  autoencoding with clustering (Bonus B
with Visualization of images and comparing it with the original.
#Part 1:

UPDATED 
What I did:
I implemented a classification model to predict art styles from the dataset. 
To handle the class imbalance (Bonus A), I used a WeightedRandomSampler. 
This method increases the sampling probability for underrepresented classes, balancing the training process.

To train the classification model, run:
python trainpart1.py

This trains the model for 20 epochs.
The trained model is saved as wikiart.pth (not included due to file size).

Test the classification model:
To evaluate the model’s performance on the test set, run:
python testtrainpart1.py
This script calculates evaluation metrics (accuracy, precision, recall, F1-score) and generates a confusion matrix saved as confusion_matrix.png.

#Results:
During training, the loss decreased from 1102.24 (epoch 1) to 5.28 (epoch 20), which shows that the model was learning effectively.
On the test set, the performance is outlined below:

Test Performance:
-Accuracy: ~20.79% (slightly better than random guessing).
-Precision: ~14.26% (model overpredicts majority classes).
-Recall: ~13.91% (struggles with correct predictions for sparse classes).
-F1-Score: ~13.13% (balance of false positives and negatives).
Impact of using WeightedRandomSampler:
While the sampler helped balance the class representation during training, the model still struggled with underrepresented classes. To improve performance, I should try adjusting the learning rate, training for more epochs, or trying upsampling techniques to better handle the sparse classes.

#Part 2: UPDATED
What I did:
For Part 2, I implemented an autoencoder in wikiart.py, which was imported and used in the training and testing scripts below. The autoencoder’s purpose was to compress images into a latent space and reconstruct them as closely as possible.

To train the autoencoder run the script below:
python trainauto.py

#UPDATED
This script trains the autoencoder for 20 epochs to minimize the Mean Squared Error (MSE) between the original and reconstructed images. 
I used Mean Squared Error (MSE) as the loss function because it checks how close the reconstructed image is to the original, pixel by pixel.
It works well for this task because it really focuses on big differences whcih pushes the autoencoder to recreate the images as closely as possible.
Once trained, the model is saved as autoencoder.pth (not included here due to file size).

Testing:
To test the autoencoder and generate the outputs for Bonus B, run:
python testtrainauto.py
This script does the following below:
-For latent encodings,  the autoencoder compresses each image into a lower-dimensional latent space.
-For dimensionality reduction, PCA is applied to reduce the high-dimensional latent space to 2D, simplifying it for visualization.
-For clustering, Agglomerative Clustering is used to group similar images based on their latent encodings

UPDATED:
Addressing Latent Representations and Clustering:
The autoencoder compressed each image into a smaller, more manageable latent representation. To make sense of this high-dimensional space, PCA was applied to reduce it to 2D for visualization. Then, Agglomerative Clustering grouped the representations into 27 clusters, matching the number of art styles in the dataset. The clustering results were visualized and saved as clusters.png.

Why did I choose PCA and Agglomerative Clustering?

PCA simplifies the latent space, which makes the visualization easy to interpret. While agglomerative clustering groups similar images into hierarchical clusters, which helps identify relationships between different art styles.

The script also generates the original vs. reconstructed images side by side and saves the result as reconstruction.png.
Output:
-Clusters:A scatter plot as clusters.png (visualizes how well the autoencoder clusters the images).
-Reconstructions: A comparison (reconstruction.png) shows original vs. reconstructed images.

Results:
Training:
During training, the loss steadily decreased from approximately 104.59 in epoch 1 to 84.49 by epoch 20. This consistent decline indicates that the autoencoder was effectively learning to compress the input images while preserving their key features.

Clustering results:
The autoencoder’s latent representations were clustered into 27 groups, corresponding to the number of art styles in the dataset. After applying PCA to reduce the latent space to 2D, the clusters were visualized in the scatter plot (clusters.png).
![image](https://github.com/user-attachments/assets/5d28e288-7f5e-4a43-9324-3ee035ad0a54)

The clusters show several distinct groupings, suggesting that the autoencoder captured style-specific features from the dataset. However, some overlap exists between clusters, likely due to shared characteristics such as color palettes or composition among certain art styles.

Reconstruction results:
The reconstructed images keep the overall structure and layout of the originals, but some fine details, like textures and small lines, are missing. This happens because the autoencoder compresses the images into a smaller space, focusing on the main features and leaving out the finer details.
![image](https://github.com/user-attachments/assets/15dd13e4-8a9c-4612-9425-134d931d5c06)

Overall the autoencoder successfully compresses images while preserving their major features. The clustering results show that the model can differentiate between art styles, even if there is some overlap. Similarly, the reconstructions demonstrate that the autoencoder captures the essential structure of images, although it struggles with finer details.









