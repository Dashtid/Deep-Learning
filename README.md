Hello and welcome to our project which is about quantifying the difference in expert segmentations from CT and MRI images.
The authors of this project is David Dashti and Filip Söderquist and we hope you enjoy running our code.


------------------------------------------------# QUBIC CHALLENGE #------------------------------------------------

Available at: https://qubiq.grand-challenge.org/Home/

--------------------------------------------------- DATA PATHS ---------------------------------------------------

The data can be found on the server using the follow path: /tf/tf/Project/Project_data

-----------------
Path for training data:   /tf/tf/Project/Project_data/training_data_v2
Path for validation data: /tf/tf/Project/Project_data/validation_data_v2
Path for test data: /tf/tf/Project/Project_data/test

-----------------

The challenge is composed of 4 different sets of images which are located in separate folders. The image sets depict 2D slices and masks of the following objects:
- brain-growth (One channel MRI, one task, seven expert masks per task)
- brain-tumor (Four channel MRI, three tasks, three expert masks per task)
- kidney (One channel CT, one task, three expert masks per task)
- prostate (One channel MRI, two task, six expert masks per task)



These can be found in respective folders in Project_data, i.e. /tf/tf/Project/Project_data/training_data_v2/brain-growth for brain-growth

------------------------------------------------ HOW TO RUN THE CODE ----------------------------------------------------

The code for the different image sets can be found using the following paths:

- brain-growth: /tf/tf/Project/Brain_Growth.py
- brain-tumor:  /tf/tf/Project/Brain_Tumor.py
- kidney:	/tf/tf/Project/Kidney.py
- prostate: 	/tf/tf/Project/Prostate.py

For brain-growth and kidney, there is only one task, but for the rest there is multiple tasks for each image set.
In order to run the different tasks, one has to manually change the variable: "task" between runs, to the corresponding task number.
It is clearly commented in the code where this has to be changed. This is due to that the server has a tendency to crash when running multiple
networks in succession. 

Valid entries for "task" are the following:

- brain-tumor:  task = 1, task = 2, task = 3
- prostate: 	task = 1, task = 2

Once the task is set, one can simply run the .py file and the network will begin training and eventually output the results.

------------------------------------------------ EXPLANATION OF THE OUTPUT -----------------------------------------------

One network will be trained for each expert mask in the image set, e.g. seven networks for brain-growth. Each network will then make predictions on the test data. The 
prediction are then added together and divided by the number of masks in order to get an ensemble. The same is done with the expert masks. The ensamble of predictions
and expert masks are then binarized and compared using nine different threshold levels. The threshold levels for binarization are the following: (0.1, 0.2,...,0.8, 0.9).
 
The result one will receive is the averaged DICE score for the task, along with an array to see how individual test images performed under each threshold level.
Each row of the array corresponds to the threshold level. Each column corresponds to a predicion ensamble for a given test image.

------------------------------------------------ MAIN FINDINGS ----------------------------------------------------------

We used the Unet architecture with the DICE loss for all our experiments. The main finding was that a deeper networks generally improved the results. The networks was made deeper by adding one convolution block toghether with one deconvolution block. Additionally, we found that the use of spatial dropout significantly outperformed
regular dropout. We hypothesise that dropping entire feature maps instead of individual nodes prevents the network from overfitting on insignificant
image features, making it more likely to learn usefull representations. Furthermore, we found that image augmentation was essential for good results. This is due to the augmentation preventing overfitting on the few number of training sample by continously providing the networks with "new" training samples. Random rotation of 10 degrees, height and width shift of 0.1 and horizontal flip was applied on all data sets, except kidney where the horizontal flip was omitted since all masks were on the right side. We also found that adding LSTM layers in the deconvolution blocks improved performance for all tasks except for brain-growth. The reason why this works is not entirely intuitive, but we hypothesise that the LSTM layers make sure that usefull image representations are saved and used in the long term. Finally, we found that for the kidney images (CT), preprocessing with intensity windowing was essential to obtain good results. 

------------------------------------------------ RESULTS ----------------------------------------------------------------

Our results for the different tasks were the following (top leaderboard result will be put in parenthesis):

Averaged DICE scores
- brain-growth:         0.9034 (0.5548)
- brain-tumor task 1:   0.8547 (0.9169)
- brain-tumor task 2:   0.7781 (0.9224)
- brain-tumor task 3:   0.7566 (0.9682)
- kidney:               0.9181 (0.8532)
- prostate task 1:      0.9296 (0.5987)
- prostate task 2:      0.9075 (0.6302)

Total avarage:          0.864  (0.7778)

------------------------------------------------ DISCUSSION ----------------------------------------------------------

The largest inconvinence of this challenge was the relatively few number of training and validation samples. Futhermore, the test data provided by the challenge
did not contain any masks. Thus, these could not be used for either training or testing, which forced us to remove a few samples from each training set 
to be used as a test set, further deminishing the already small training sets. Nevertheless, we still managed to get decent results. Though, this also means that our results are not directly comparable to the ones provided on the website, since these are based on a different dataset. However, after close inspection of the test images provided by the challenge, we can conclude that these resemble the training data to a large extent. Provided that the masks do aswell, we are confident that our results are representable, we cannot however be entirly sure since these were not given. The largest deviation compared to the leaderboard is the brain-tumor tasks. We are not entierly sure why since we cannot se the other contestants implementation, but we found that the validation set was very unlike the training set and thus made it hard to optimise the network. Perhaps the official test set was easier than the one we used, and therefore gave the contestants better results than us, or perhaps their algorithm handled four channel MRI images better than our.

Compared to the leaderboard we did really well, especially for . Unfortunatly we did not have the opportunity to enter into the challenge as we believe that we would have a good chance of making the top 3.
