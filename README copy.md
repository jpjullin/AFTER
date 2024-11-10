![After Logo](/docs/after.jpeg)

# AFTER: Audio Features Transfer and Exploration in Real-time

__AFTER__ is a diffusion-based generative model that generate audio based on two targets : one audio stream that will set the global style or timbre, and one structure input -either audio or MIDI- that sets the time-varying features of the signal. 

This repository is a real-time implementation of the research paper _Combining audio control and style transfer using latent diffusion_ ([link](https://arxiv.org/abs/2408.00196)) by Nils Demerlé, P. Esling, G. Doras and D. Genova. The code to train a non-streamable version of the model is available on this [github repository](https://github.com/NilsDem/control-transfer-diffusion/), and you can listen to some transfer examples on the original paper supporting [webpage](https://nilsdem.github.io/control-transfer-diffusion/). Our real-time implementation of the model runs in MaxMSP and AbletonLive thanks to [_nn_tilde_](https://github.com/acids-ircam/nn_tilde), an external that embeds Pytorch models into MaxMSP. 

We will be releasing the code to train your own streambable version of the model very soon. In the meantime, you can try and experiment with three of our pretrained models !

## Installation

The only requirement for inference is the [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) external.

Clone this repository to a local directory, and make sure you add it to the list of Max paths (Options -> File Preferences...) with the recursive subfolders option enabled. Download the pretrained model with the links in the below and copy them in the same directory.  

## MIDI-to-audio 

Our MIDI-to-audio model is a 4 voice polyphonic synth that generates audio for pitch and velocity as well as a timbre target defined as follows: 
- __Audio based__: using the method __foward__, timbre is extracted from an audio stream (with a receptive field for around 3seconds). For demonstration we provide a few samples coming from the training set.
- __Manual exploration__: using __foward_manual__, timbre is specified through 8 sliders that set a position in a learned 8-dimensional timbre space. 

The guidance parameter sets the conditioning strength on the MIDI input, and the number of diffusion steps improves generation quality at the cost of higher CPU load.

You can download our model trained on the [SLAKH dataset](http://www.slakh.com/) at the following [link](https://nubo.ircam.fr/index.php/s/tHMmFmkF6kgn7ND/download).



Audio timbre target           |  Manual timbre control
:-------------------------:|:-------------------------:
<img src="docs/midi_to_audio.png"   height="500"/>| <img src="docs/midi_to_audio_manual.png"  height="500"/>






## Audio-to-audio 

In audio-to-audio mode, time-varying features are extracted from an audio audio stream and transferred to the timbre of a second audio stream. Similarly, the guidance parameter sets the conditioning strengh on the structure input, and the number of diffusion steps improves generation quality at the cost of higher CPU load.

You can download our model trained on the [SLAKH dataset](http://www.slakh.com/) at the following [link](https://nubo.ircam.fr/index.php/s/NCHZ5Q9aMsFxmyp/download).


<img src="docs/audio_to_audio.png"  height="500"/>



# Artistic applications

This model has been used in three artistic projects :
- [_The Call_](https://www.serpentinegalleries.org/whats-on/holly-herndon-mat-dryhurst-the-call/) by Holly Herndon and Mat Dryhurst. Interactive sound installation with singing voice transfer at Serpentine Gallery in London until February 2nd, 2025. 
- Live performance by french electronic artist Canblaster for Forum Studio Session at IRCAM. Full concert available on [youtube](https://www.youtube.com/watch?v=0E9nNyz4pv4).
- [Nature Manifesto](https://www.centrepompidou.fr/fr/programme/agenda/evenement/dkTTgJv), an immersive sound installation by Björk and Robin Meier. At Centre Pompidou in Paris from November 20 to December 9, 2024. 

Looking forward to hear your own contributions and artistic use of the model, stay tuned for the training code release ! 



