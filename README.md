# FNet

## About
The PyTorch implementation of FNet from the paper [*FNet: Mixing Tokens with Fourier Transforms*](<https://arxiv.org/abs/2105.03824>).

## Citation
```
@inproceedings{lee-thorp-etal-2022-fnet,
  title     = {FNet: Mixing Tokens with Fourier Transforms},
  author    = {Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {07},
  year      = {2022},
  publisher = {Association for Computational Linguistics},
  pages     = {4296--4313}
}
```

## Datasets
1. LRA: https://mega.nz/file/tBdAyCwA#AvMIYJrkLset-Xb9ruA7fK04zZ_Jx2p7rdwrVVaTckE

## Training Steps
1. Create a data folder:
```console
mkdir data
```

2. Download the dataset compressed archive
```console
wget $URL
```

3. Decompress the dataset compressed archive and put the contents into the data folder
```console
unzip $dataset.zip
mv $datast ./data/$datast
```

4. Run the main file
```console
python $dataset_main.py --task="$task"
```

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt