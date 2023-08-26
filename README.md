# UDA_Med_Landmark

Code of MICCAI 2023 paper: "Unsupervised Domain Adaptation for Anatomical Landmark Detection".

## Installation
1. Install Python3 and PyTorch >= v1.3
2. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/UDA_Med_Landmark.git
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
## Datasets
* Head: Source domain of cephalometric landmark detection ([Download Link](https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466590)). Put the downloaded `RawImage` and `AnnotationsByMD` under `data/Head`.
* HeadNew: Target domain of cephalometric landmark detection ([Download Link](http://vision.seecs.edu.pk/CEPHA29/)). Put the downloaded `Cephalograms` and `Cephalometric_Landmarks` under `data/HeadNew`.
* JSRT: Source domain of lung landmark detection ([Download Link](http://db.jsrt.or.jp/eng.php)). 
* MSP: Target domain of lung landmark detection, which consists of three datasets: Montgomery ([Download Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)), Shenzhen ([Download Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)), and PadChest ([Download Link](https://bimcv.cipf.es/bimcv-projects/padchest/)).

## Training
