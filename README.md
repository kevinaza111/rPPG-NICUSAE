# rPPG-NICUSAE

## Environment configuration
'''
 install Pytorch 2.1.2 and torchvision 0.16.2
'''
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
'''
Relevant dependencies have been described in requirement.txt, just install them as required.
'''
conda install --yes --file requirements.txt
'''

## Data Preparation
We use UBFC, NBHR, NICU-NPS．NICU-NPS will be made public after the paper is published. NBHR and UBFC are public databases.

We provide a few data samples for you to test demo.

## Training of adaptive lighting enhancement module
You need to configure the data_root and save_path parameters in the clip_train.py file. After executing the training, you can get the adaptive lighting enhancement model parameter file in the save_path path. This model file will be used in end_to_end.py

## Use of test demo
Place the image collection of video data in the ./data directory, and then execute the end_to_end.py file. The image collection will first undergo adaptive lighting enhancement, and then obtain the rPPG signal through our rPPG signal extraction network, and then pass through an FFT Get HR signal.

# The rest of the code will be coming soon
