# Deep Learning for Medical Image Segmentation and Classification

Welcome! This project explores the quantification of differences in expert segmentations from CT and MRI images using deep learning. The authors are **David Dashti** and **Filip Söderquist**.

---

## QUBIQ Challenge

This work is based on the [QUBIQ Challenge](https://qubiq.grand-challenge.org/Home/), which provides multi-expert annotated datasets for several medical imaging tasks.

---

## Data Organization

All data is located on the server at `/tf/tf/Project/Project_data`:

- **Training data:** `/tf/tf/Project/Project_data/training_data_v2`
- **Validation data:** `/tf/tf/Project/Project_data/validation_data_v2`
- **Test data:** `/tf/tf/Project/Project_data/test`

The challenge includes four datasets, each in its own folder, containing 2D slices and masks for:

- **brain-growth:** One-channel MRI, one task, seven expert masks per task
- **brain-tumor:** Four-channel MRI, three tasks, three expert masks per task
- **kidney:** One-channel CT, one task, three expert masks per task
- **prostate:** One-channel MRI, two tasks, six expert masks per task

Example: `/tf/tf/Project/Project_data/training_data_v2/brain-growth` contains the brain-growth data.

---

## How to Run

Scripts for each dataset are provided:

- **brain-growth:** `/tf/tf/Project/Brain_Growth.py`
- **brain-tumor:** `/tf/tf/Project/Brain_Tumor.py`
- **kidney:** `/tf/tf/Project/Kidney.py`
- **prostate:** `/tf/tf/Project/Prostate.py`

Some datasets have multiple tasks. To run a specific task, set the `task` variable in the script to the desired task number (see comments in the code). This manual step is required to avoid server crashes when running multiple networks in succession.

Valid `task` values:

- **brain-tumor:** 1, 2, 3
- **prostate:** 1, 2

After setting the task, run the corresponding `.py` file to train the network and generate results.

---

## Output Explanation

- For each expert mask, a separate network is trained.
- Each network predicts on the test data.
- Predictions are ensembled (averaged) and compared to ensembled expert masks.
- Binarization is performed at nine thresholds: 0.1, 0.2, ..., 0.9.
- Results include the average DICE score per task and a matrix showing individual test image performance at each threshold.

---

## Main Findings

- **Architecture:** U-Net with DICE loss was used throughout.
- **Depth:** Deeper networks (more convolution/deconvolution blocks) improved results.
- **Dropout:** Spatial dropout outperformed standard dropout, likely by encouraging the network to learn more robust features.
- **Augmentation:** Essential due to limited data. Used random rotation (10°), width/height shift (0.1), and horizontal flip (except for kidney).
- **LSTM Layers:** Adding LSTM layers in deconvolution blocks improved performance (except for brain-growth).
- **Preprocessing:** Intensity windowing was crucial for kidney (CT) images.

---

## Results

Average DICE scores (top leaderboard result in parentheses):

- **brain-growth:** 0.9034 (0.5548)
- **brain-tumor task 1:** 0.8547 (0.9169)
- **brain-tumor task 2:** 0.7781 (0.9224)
- **brain-tumor task 3:** 0.7566 (0.9682)
- **kidney:** 0.9181 (0.8532)
- **prostate task 1:** 0.9296 (0.5987)
- **prostate task 2:** 0.9075 (0.6302)

**Total average:** 0.864 (0.7778)

---

## Discussion

- The main challenge was the small number of training/validation samples.
- Test data did not include masks, so some training samples were held out for testing, further reducing training size.
- Results are not directly comparable to the leaderboard, as the test sets differ.
- The largest deviation was in brain-tumor tasks, possibly due to differences in data distribution or handling of multi-channel MRI.
- Despite these challenges, our results are strong, especially for kidney and prostate.

---

## How to Cite

If you use this code or results, please cite this repository and the QUBIQ Challenge.

---

## Contact

For questions or suggestions, please contact **David Dashti** or **Filip Söderquist**.
