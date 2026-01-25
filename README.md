

# DeepMapper: attention-based autoencoder for system identification in wound healing and stage prediction

This code is an implementation of our paper: "[DeepMapper: attention-based autoencoder for system identification in wound healing and stage prediction](https://www.biorxiv.org/content/10.1101/2024.12.17.628977v1.abstract). Lu, F., Zlobina, K., Osorio, S., Yang, H.Y., Nava, A., Bagood, M.D., Rolandi, M., Isseroff, R.R. and Gomez, M." 

It uses an attentention based autoencoder for identifying a linear dynamics for a nonlinear wound healing system.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Our code is based on Python 3.11, Pytorch 2.7, numpy, pandas and etc.

## Training

To train the model(s) in the paper, run this command:

```train
python main.py
```

> 📋 You can modifiy the network structure inside the code.

## Evaluation

To reproduce the plots in our paper, simply go to the **jupter** folder otebook DeepMapper.ipnb and run each code block.

We also provide the source code for HealNet in the **jupyter** folder, one baseline we used for the paper.


## Contributing

> 📋 All content is licensed under the MIT license.