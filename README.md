# TSCurator

This repo is the official Pytorch implementation of TSCurator: "[TSCurator: Exploring Good Data for Long-term Time Series Forecasting]". 


## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n TSCurator python=3.9
conda activate TSCurator
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. Additionally, the datasets mentioned in the Appendix can be found at [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval). All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
```
sh scripts/generate_indicator.sh
sh scripts/ETTh2_ours.sh
```

Additionally, Timer and Moirai in the Appendix can be trained in the OpenLTM folder.
