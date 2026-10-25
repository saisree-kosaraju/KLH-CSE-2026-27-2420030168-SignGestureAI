# AI-Based Hand Gesture Recognition for Sign Language Communication

A deep learning-based Indian Sign Language (ISL) recognition system that uses MediaPipe landmarks, engineered geometric features, temporal convolutional networks, and Head-Token Mixing Attention (HTMA) to recognize signs from video and convert the predicted sign into text and speech.

## 📌 Project Overview

Communication can be challenging when sign language users interact with people who do not understand sign language. This project addresses that challenge by developing an AI-based system for recognizing isolated Indian Sign Language signs from video.

The system uses the **INCLUDE dataset** and a custom landmark-based deep learning architecture. It extracts hand, body pose, and selected facial landmarks, combines them with geometric and motion features, and models temporal relationships to classify signs into 262 classes.

The predicted sign is displayed as text and can be spoken through browser-based text-to-speech.

## 🎯 Objectives

- Recognize Indian Sign Language signs from video input.
- Extract hand, pose, and facial landmarks using MediaPipe.
- Capture geometric relationships and motion patterns between landmarks.
- Learn temporal features using part-aware convolutional encoders and attention.
- Handle missing landmark detections through availability-aware feature fusion.
- Classify signs across 262 categories from the INCLUDE dataset.
- Provide predicted sign labels as text and speech through a web application.

## 🏗️ System Architecture

The recognition pipeline consists of the following stages:

1. **Video Input:** Accept a sign video through upload or camera recording.
2. **Frame Preprocessing:** Use OpenCV to sample 64 frames uniformly from each video. Short videos are padded using their last frame.
3. **Landmark Extraction:** Use MediaPipe to extract features for the left hand, right hand, body pose, and selected facial landmarks.
4. **Geometric Feature Engineering:** Calculate 469 geometric and motion-related features, along with a 4-dimensional availability mask indicating detected body parts.
5. **Part-Aware Temporal Encoding:** Apply separate temporal Conv1D encoders to the landmark streams.
6. **Gated Feature Fusion:** Combine the landmark, geometry, and availability representations using learned gates to produce a 320-dimensional feature vector for each frame.
7. **Temporal Attention:** Process the sequence using four Head-Token Mixing Attention (HTMA) blocks.
8. **Sign Classification:** Combine the CLS representation with attention pooling and classify the video into one of 262 sign categories.
9. **Text and Speech Output:** Display the predicted sign label and generate spoken output using the browser Web Speech API.

## 🧠 Proposed Model

The custom recognition model is named **IncludePartAwareConvHTMAV2**.

Its main components are:

- **MediaPipe landmark extraction:** Produces hand, pose, and selected facial landmark features.
- **Engineered geometry features:** Represents joint relationships, bone lengths, angles, spatial relationships, and motion information.
- **Part-aware temporal Conv1D encoders:** Learn local temporal patterns from separate body-part streams.
- **Availability-aware gated fusion:** Learns how to combine feature streams while accounting for missing detections.
- **Head-Token Mixing Attention (HTMA):** Models temporal relationships across the sampled frames.
- **CLS and attention pooling:** Summarizes the video sequence for classification.
- **262-class classifier:** Predicts the corresponding sign label.

Unlike RGB-based architectures that use image CNN backbones such as EfficientNet or MobileNet, this implementation uses custom temporal Conv1D layers on landmark and geometry sequences. MediaPipe provides the pretrained landmark extraction component.

## 📚 Dataset

**INCLUDE — Indian Sign Language Dataset**

The project uses the official INCLUDE data partition, with the `secondnumber` class excluded from the original label space.

| Split | Videos |
|---|---:|
| Training | 3,121 |
| Validation | 347 |
| Testing | 816 |
| Total | 4,284 |

- **Number of classes:** 262
- **Sign language:** Indian Sign Language (ISL)
- **Input representation:** MediaPipe landmarks, geometric features, and availability indicators
- **Frame sequence length:** 64 frames per video

The test set was kept separate for final evaluation.

Dataset resource: [AI4Bharat INCLUDE Repository](https://github.com/AI4Bharat/INCLUDE)

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing, model training, and inference |
| PyTorch | Custom neural network, training, checkpointing, and evaluation |
| MediaPipe | Hand, pose, and facial landmark extraction |
| OpenCV | Video reading and frame preprocessing |
| NumPy | Numerical operations and feature arrays |
| Pandas | Dataset metadata and label mapping |
| FastAPI | Backend inference API |
| React | User interface |
| Vite | Frontend development and build tooling |
| Web Speech API | Browser-based speech output |
| Git and GitHub | Version control and project hosting |

## 📊 Results and Performance Evaluation

The final model was evaluated using the best-validation exponential moving average (EMA) checkpoint from epoch 38 on the locked INCLUDE test set.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **98.04%** |
| Top-5 Accuracy | **99.51%** |
| Macro-F1 Score | **98.09%** |
| Balanced Accuracy | **98.19%** |
| Test Videos | 816 |
| Classes Evaluated | 262 |

These results describe the reported evaluation on the INCLUDE test set. They should not be interpreted as measured webcam accuracy or as proof of superiority over models evaluated under different datasets and protocols.

## 🌐 Web Application

The project includes a web application built with React and Vite, connected to a FastAPI inference backend.

The application supports:

- Video upload and camera recording.
- Sign prediction through the trained model.
- Display of the predicted label and confidence.
- Ranked Top-5 predictions.
- Browser-based speech output using the Web Speech API, with an Indian-English voice preference when available.

## 🚀 Future Enhancements

- Evaluate and improve robustness on unseen signers and different recording conditions.
- Measure webcam performance separately from the offline test set.
- Optimize inference for lower latency and resource-constrained devices.
- Extend the system from isolated sign classification to continuous sign recognition.
- Investigate sentence-level recognition and translation.
- Conduct component ablation studies to quantify the contribution of geometry, gating, part-aware encoders, and HTMA.

## 👩‍💻 Author

**K. Sai Sree Chowdary**  
**Roll Number:** 2420030168  
**Institution:** KLH University


## 🙌 Acknowledgements

We acknowledge the AI4Bharat INCLUDE dataset and the open-source communities behind MediaPipe, PyTorch, OpenCV, FastAPI, and React for supporting the development of this project.
