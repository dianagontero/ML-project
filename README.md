
# Brain Tumor Classification - ML Project

## Authors
- Nicolo Vacis, Giovanni Vaccarino, Vittorio Palladino, Diana Gontero  
Department of Computer Science, University of Illinois Chicago

## Description
This project investigates the application of deep learning techniques for automatic brain tumor classification using magnetic resonance imaging (MRI). The addressed problem is a four-class classification: glioma, meningioma, pituitary tumors, and healthy brain images.

Two main approaches were developed:
- **CNN from scratch**: a convolutional neural network trained from the ground up.
- **Transfer Learning with MobileNetV2**: a model pre-trained on ImageNet, fine-tuned on the brain tumor dataset.

Additionally, Grad-CAM was used to enhance model interpretability by visualizing the relevant anatomical regions driving predictions.

## Dataset
- **Source**: [Kaggle MRI Brain Tumor Dataset](https://www.kaggle.com/datasets/alaminbhuyan/mri-image-data/data)
- **Classes**: glioma, meningioma, pituitary tumor, no tumor
- **Format**: MRI images in train/test split (NPZ)
- **Preprocessing**: normalization according to ImageNet parameters, stratified train/val split, class balancing
- **Augmentations**: random horizontal flip, random rotations up to 20°, only during training

## Machine Learning Task
- **Problem**: multi-class classification of MRI images
- **Metrics**: recall (priority), precision, F1-score, accuracy, confusion matrix

## Architecture and Techniques
- **CNN from scratch**: stack of convolutions, ReLU, max-pooling
- **MobileNetV2**: pre-trained backbone, fine-tuning
- **Optimization**: Adam, cross-entropy, initial learning rate 1e-5, StepLR scheduler, early stopping
- **Interpretability**: Grad-CAM to visualize model attention regions

## Results
- **MobileNetV2** achieved 95% accuracy, higher than the 88% of the CNN from scratch
- Significant improvements on challenging classes (glioma, meningioma)
- Data augmentation was fundamental for generalization
- Grad-CAM shows the model focuses on clinically relevant regions

## Lessons Learned
- Transfer learning is crucial with limited datasets
- Optimization choices (scheduler, early stopping) prevent overfitting
- Interpretability via Grad-CAM increases clinical trust

## Project Structure
- `BrainTumorClassification_FromScratch.ipynb`: CNN training from scratch
- `BrainTumorClassification.ipynb`: transfer learning with MobileNetV2
- `Heatmap_BrainTumorClassification.ipynb`: Grad-CAM visualization
- `FinalReport_MLProject.pdf`: full report

## Main Dependencies
- Python, PyTorch, torchvision, scikit-learn, matplotlib, seaborn, tqdm, Google Colab

## Getting Started
1. Upload the data to the appropriate folder on Google Drive
2. Open the notebooks in Google Colab
3. Follow the instructions in the notebooks for training, evaluation, and visualization

## References
- See the "References" section in the PDF report for the full bibliography.
