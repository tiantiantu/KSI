# KSI framework
This repository contains codes for Knowledge Source Intergration (KSI) framework in the paper
* **Bai, T., Vucetic, S., Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources, The Web Conference (WWW'19), 2019.**

I used the following environment for the implementation:
* python==3.7.0
* torch==0.4.1
* numpy==1.15.1
* sklearn==0.19.2

Before running the program, you need to apply for [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset and put two files "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv" under the same folder of the project.

Once you get these two files, run preprocessing scripts "preprocessing1.py", "preprocessing2.py", "preprocessing3.py" in order.

After running three preprocessing files, you can run any of four models ("KSI_LSTM.py", "KSI_LSTMatt.py", "KSI_CNN.py", "KSI_CAML.py") to see how much improvement KSI framework brings to the specific model.
