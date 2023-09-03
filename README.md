# UDA_Med_Landmark

Code of MICCAI 2023 paper: "Unsupervised Domain Adaptation for Anatomical Landmark Detection".

## Installation
1. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/UDA_Med_Landmark.git
```
2. Create a new conda environment.
```Shell
conda create -n uda_med_landmark python=3.9
conda activate uda_med_landmark
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
## Datasets Preparation
* **Head**: Source domain of cephalometric landmark detection ([Download Link](https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466590)). Put the downloaded `RawImage` and `AnnotationsByMD` under `data/Head/`.
* **HeadNew**: Target domain of cephalometric landmark detection ([Download Link](http://vision.seecs.edu.pk/CEPHA29/)). If the link is not available, please read the Section 4 "Usage Notes" of paper (https://arxiv.org/pdf/2302.07797.pdf) for data downloading. Put the downloaded `Cephalograms` and `Cephalometric_Landmarks` of the training set under `data/HeadNew/`.
* **JSRT**: Source domain of lung landmark detection ([Download Link](http://db.jsrt.or.jp/eng.php)). Put the downloaded `All247images` under `data/JSRT/`. Collect the landmark annotations from [HybridGNet](https://github.com/ngaggion/HybridGNet/tree/main) and put them under `data/JSRT/annos/`. Run the `preprocess.py` under `data/JSRT` to generate `Images`. 
* **MSP**: Target domain of lung landmark detection, which consists of three datasets: Montgomery ([Download Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)), Shenzhen ([Download Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)), and Padchest ([Download Link](https://bimcv.cipf.es/bimcv-projects/padchest/)). **1)** For **Montgomery**, put the downloaded `CXR_png` under `data/Montgomery/`. Collect the landmark annotations from [HybridGNet](https://github.com/ngaggion/HybridGNet/tree/main) and put them under `data/Montgomery/annos_RL/` and `data/Montgomery/annos_LL/`. Run the `preprocess.py` under `data/Montgomery` to generate `Images`; **2)** For **Shenzhen**, put the downloaded `CXR_png` under `data/Shenzhen/`. Collect the landmark annotations from [HybridGNet](https://github.com/ngaggion/HybridGNet/tree/main) and put them under `data/Shenzhen/annos_RL/` and `data/Shenzhen/annos_LL/`. Run the `preprocess.py` under `data/Shenzhen` to generate `Images`; **3)** For **Padchest**, download the landmark annotations from [here](https://drive.google.com/file/d/15qdzekQfj4zgkVgfi_x1WlKAq2wITl8i/view?usp=drive_link), unzip it and put it under `data/Padchest/annos`. Then select those images with landmark annotations and put them under `data/Padchest/Images/`.

You will have the following structure:
````
UDA_Med_Landmark
-- data
   |-- Head
       |-- RawImage
       |-- AnnotationsByMD
   |-- HeadNew
       |-- Cephalograms
       |-- Cephalometric_Landmarks
       |-- img2size.json
       |-- img2dist.json
       |-- train_list.txt
       |-- test_list.txt
   |-- JSRT
       |-- All247images
       |-- preprocess.py
       |-- Images
       |-- annos
   |-- Montgomery
       |-- CXR_png
       |-- preprocess.py
       |-- Images
       |-- annos_RL
       |-- annos_LL
       |-- train_list.txt
       |-- test_list.txt
   |-- Shenzhen
       |-- CXR_png
       |-- preprocess.py
       |-- Images
       |-- annos_RL
       |-- annos_LL
       |-- train_list.txt
       |-- test_list.txt
   |-- Padchest
       |-- Images
       |-- annos
       |-- train_list.txt
       |-- test_list.txt
````

## Training
Take cephalometric landmark detection as example.
1. Go to folder `lib`, run `preprocess.py Head` and `preporcess.py HeadNew` to preprocess the two datasets, respectively.
2. Back to folder `UDA_Med_Landmark`, configure the command in `run_train.sh` as needed, then run `bash run_train.sh` to start training.

## Testing
Take cephalometric landmark detection as example.
1. Preprocess `Head` and `HeadNew` the same way as in training.
2. Back to folder `UDA_Med_Landmark`, configure the command in `run_test.sh` as needed, then run `bash run_test.sh` to start testing.

**Trained model weights**:
* [cephalometric landmark detection model](https://drive.google.com/file/d/1wGgQgdpdvyINNN7hvT49TLFjuxruRdRR/view?usp=drive_link)
* [lung landmark detection model](https://drive.google.com/file/d/1vaHjNunUD6Uv4X4H7ve1huXDKRoUCQz2/view?usp=drive_link)
