![After Logo](/after.jpeg)

# AFTER: Audio Features Transfer and Exploration in Real-time

Official repository of AFTER, a real-time implementation of : _Combining audio control and style transfer using latent diffusion_ ([article link](https://arxiv.org/abs/2408.00196)) by Nils Demerlé, P. Esling, G. Doras and D. Genova.

_AFTER_ is a diffusion-based generative model that generate audio based on two targets : one audio stream that will set the global style or timbre, and one structure input -either audio or MIDI- for the time-varying features of the signal. 

The code to train a non-streamable version model on this [github repository](https://github.com/NilsDem/control-transfer-diffusion/), and you can listen to some transfer examples on the orignal paper supporting [webpage](https://nilsdem.github.io/control-transfer-diffusion/).

Thanks to the use of  [_cached_convolution_](https://github.com/acids-ircam/cached_conv) and the [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) library we developped a real-time version of the model that runs in MaxMSP and AbletonLive.

We will be releasing the code to train your own streambable version of the model very soon. In the meantime, you can try and experiment with three of our pretrained models in MaxMSP ! 


## General work  



## MIDI-to-audio 










