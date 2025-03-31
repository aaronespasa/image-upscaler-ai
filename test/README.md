# Set up the environment

## Install the dependencies

Install Python 3.10. Python 3.13 has incompatibility issues with the basicsr package.

```bash
conda create --name super-resolution python=3.10 -y
conda activate super-resolution
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Download the DRCT

### Download the `DRCT` PyTorch code

This is the code for the DRCT model: https://github.com/ming053l/DRCT/tree/main/drct

You can download this folder in Windows using the command:

```bash
.\clone_dcrt.ps1
```

### Download the `Real_DCRT-L_GAN_SRx4` model

Download the files `net_d_latest.pth` and `net_g_latest.pth` from the following link:
https://drive.google.com/drive/folders/1W-2EEC5mclFzzWrp65u7JDuLDiaA_vPX

Put the downloaded files in the `models` folder.

Credits to:
- DRCT [[LICENSE](https://github.com/ming053l/DRCT/blob/main/LICENSE)]: https://github.com/ming053l/drct