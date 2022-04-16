This repository provides the functions and notebooks which were used to perform test and evaluation for the paper "Independent assessment of a deep learning system for lymph node metastasis detection on the Augmented Reality Microscope". Note that the code provided here is only for reference; it will not work without the corresponding data and ground truth files.

Although the actual whole slide images and fields of view are not provided, the data folder contains a high-level representation of the entire test set in .csv format, including addressable locators for individual regions of interest and fields of view. For each annotated region of interest, the corresponding line within the csv contains its subclass, ground truth label, and model inference. 