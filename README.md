# BabyBeacon Emotion Recognition
## BabyBeacon Overview
In today’s fast-paced society, parents often balance numerous responsibilities including work, errands and childcare, leading to increased stress and divided attention, especially during travel. The presence of a baby in the backseat presents an added level of distraction as parents feel compelled to monitor their child's behaviour, increasing the risk of an accident. Existing baby monitoring devices primarily focus on home environments and are not catered to the variability of a car's environment. Our project called ‘BabyBeacon’ aims to address this issue by implementing an in-car system that continuously monitors the baby’s behaviour and provides real-time alerts to the driver, allowing them to focus on the road. The potential end users of our system are individuals who frequently travel with infants or toddlers. This includes: parents, guardians, caregivers and child safety workers. The system will incorporate a wide range of technical concepts, including facial/movement recognition, sound detection, and a user interface, along with an API to facilitate data storage and machine learning integration.
Additionally, testing will be conducted under various driving conditions to validate both the performance and response time of the system. The funding for the project will be provided by the McMaster ECE Capstone budget. The primary risk is the system’s ability to accurately interpret the baby’s sounds and motion in the vehicle, as incorrect suggestions may further distract the driver. In conclusion, the project aims to reduce driver distraction and place their mind at ease by ensuring the needs and safety of the baby are monitored. 

## Repository Overview
This repository is a component of the Baby Beacon Project. It deals with recognizing different sounds a baby could make during a car ride. Using Tensorflow and Keras in Python, we have developed a model that would be able to recognize the following emotions based on different cries. 

- Belly Pain
- Needs Burping
- Discomfort
- Hungry
- Tired

## Training the AI Model
During the research phase, we found a few pre-trained models that work for Emotion Recognition in adults, but not many for infants. 
For Adult Emotion Recognition these models and datasets could be useful:
- [Github.com Voice Emotion Detector](https://github.com/crhung/Voice-Emotion-Detector)

### Dataset
After further research, we came across another GitHub Repository that has datasets for infants' emotions. 

[Medium.com](https://medium.com/@rtsrumi07/deep-learning-for-classifying-audio-of-infant-crying-with-hands-on-example-a01d3cbf0f74) article by Rabeya Tus Sadia helped us find the dataset [Donate-A-Cry Corpus](https://github.com/gveres/donateacry-corpus) Dataset. Although using this dataset helped us get a model, it wasn't very accurate. While testing, the model would only recognize inputs as 'hungry'. 

We found another dataset that uses [Donate-A-Cry Corpus](https://github.com/gveres/donateacry-corpus) as a skeleton and added more emotions and synthetic data. [BabyCry](https://github.com/martha92/babycry?utm_source=chatgpt.com) is the final dataset we used to create our model.

As stated in [BabyCry](https://github.com/martha92/babycry?utm_source=chatgpt.com): 

- _audio files contain baby cry samples of infants from 0 to 2 years old_
- _corressponding tagging information (the suspected reasons of cry) encoded in the file names_
- _converted all files to WAV file format so that it could be easily read and interconverted by Python audio libraries (librosa, Wave and SoundFile)_

> Data Augmentation Our objective was to balance all the 9 classes, by creating new synthetic training samples by adding small perturbations on our initial training set, so that the model is not biased towards any one single class thus enhance its ability to generalize.

We created two models with two datasets:
1. This dataset uses ONLY the Augmented Data found in the [BabyCry](https://github.com/martha92/babycry?utm_source=chatgpt.com)  repository
2. This dataset uses a combination of the Augmented Data from [BabyCry](https://github.com/martha92/babycry?utm_source=chatgpt.com) and the actual data from [Donate-A-Cry Corpus](https://github.com/gveres/donateacry-corpus)

Each dataset gave us two models in .keras and .h5 file formats, with varying results, which will be discussed later. 
