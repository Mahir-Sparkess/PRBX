This is project repository for the PRBX module. Submitted in part fulfilment for the degree of MEng.

# Inverse Modelling of Woven Fabric with Generative Adversarial Networks

by Mahir Rahman.

----

Download the dataset used in the project here: https://drive.google.com/file/d/1--vl5kw8TsqI6VcQU4qqvegETBD1hwMK/view?usp=share_link

Download the weights for the Generator here: https://drive.google.com/file/d/1S2-cspF4YLWT-uNxGNjDMkU-hOc-5rTQ/view?usp=share_link

Download the weights for the Discriminator here: https://drive.google.com/file/d/1ndfadmiqKMzODYZbFUu6SuXcqn0BYv9Z/view?usp=share_link

To run the model, follow the steps:

1. Create a conda environment with the required packages with `$ conda create --name PRBX --file requirements.txt`
2. To test the model, run `test.py` with the current saved model weights in the file.
3. To train the model, run `train.py` with the available dataset to start initialisation training.
4. For rendered training, the `train.py` can be modified with the rendered discriminator and modifying the generator to output a rendered texture as a tensor.

----

**Note:** You may require changing the filepaths of the constants in `train.py` and `test.py`
