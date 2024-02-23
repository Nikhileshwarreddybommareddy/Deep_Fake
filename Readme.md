# Deepfake Detection with CelebrityDF Dataset

## Introduction

In the digital era, the surge of deepfake videos has emerged as a significant challenge. Our project utilizes the CelebrityDF dataset, featuring 6,000 real and fake celebrity videos, to develop models capable of detecting deepfakes. By applying computer vision and deep learning techniques, such as facial landmark extraction and movement analysis, combined with transfer learning, we aim to differentiate between authentic and manipulated videos. Our goal is to establish a reliable deepfake detection system that adapts to evolving manipulation techniques, ensuring video content authenticity.

## Dataset

The Celeb-DF dataset, comprising real and synthetic videos of celebrities, serves as the foundation for our algorithm development. Its diverse video content, including variations in lighting, expressions, and angles, enhances our model's effectiveness in various conditions.

## Methodology

- **Face Detection:** Utilizing a CNN model for accurate face detection, processing video frames, and saving detected faces for further analysis.
- **Class Imbalance Handling:** Equalizing the number of frames from original and fake videos to mitigate class imbalance.
- **Model Selection:** Experimenting with several transfer learning models (InceptionV3, DenseNet201, Xception, etc.) to identify the most effective for deepfake detection.
- **Data Management:** Implementing custom data generators for efficient batch processing and employing data augmentation to balance the dataset.
- **Model Optimization:** Exploring transfer learning strategies, with a focus on freezing vs. unfreezing convolutional layers to enhance model accuracy.

## Results

Our methodology, from face detection to model optimization, culminated in identifying DenseNet as the best-performing model, achieving a significant improvement in detection accuracy.

## Conclusion

This project underscores the importance of advanced deep learning techniques in combating the proliferation of deepfake videos. Through our comprehensive approach, we've made strides towards creating a robust deepfake detection system, ensuring the integrity of digital content in today's fast-paced digital landscape.
