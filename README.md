This is project repository for the PRBX module. Submitted in part fulfilment for the degree of MEng.

# Inverse Modelling of Woven Fabric with Generative Adversarial Networks

by Mahir Rahman.

----

To run the model, follow the steps:

1. Create a conda environment with the required packages with `$ conda create --name PRBX --file requirements.txt`
2. To test the model, run `test.py` with the current saved model weights in the file.
3. To train the model, run `train.py` with the available dataset to start initialisation training.
4. For rendered training, the `train.py` can be modified with the rendered discriminator and modifying the generator to output a rendered texture as a tensor.
