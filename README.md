![After Logo](/docs/after.jpeg)

# AFTER: Audio Features Transfer and Exploration in Real-time

__AFTER__ is a diffusion-based generative model that generate audio based on two targets : one audio stream that will set the global style or timbre, and one structure input -either audio or MIDI- that sets the time-varying features of the signal. 

This repository is a real-time implementation of the research paper _Combining audio control and style transfer using latent diffusion_ ([article link](https://arxiv.org/abs/2408.00196)) by Nils Demerlé, P. Esling, G. Doras and D. Genova. The code to train a non-streamable version of the model is avaialble on this [github repository](https://github.com/NilsDem/control-transfer-diffusion/), and you can listen to some transfer examples on the orignal paper supporting [webpage](https://nilsdem.github.io/control-transfer-diffusion/). Our real-time implementation of the model runs in MaxMSP and AbletonLive thanks to [_nn_tilde_](https://github.com/acids-ircam/nn_tilde), a MaxMSP external that embeds Pytorch models. 

We will be releasing the code to train your own streambable version of the model very soon. In the meantime, you can try and experiment with three of our pretrained models !

## Requirements

For inference, you only need to install the [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) external.

## MIDI-to-audio 

Our MIDI-to-audio model is a 4 voice polyphonic synth that generates audio for pitch and velocity as well as a timbre target defined as follows: 
- __Audio based__: using the method __foward__, timbre is extracted from an audio stream (with a receptive field for around 3seconds). For demonstration we provide a few samples coming from the training set.
- __Manual exploration__: using __foward_manual__, timbre is specified through 8 sliders that set a position in a learned 8-dimensional timbre space. 

The guidance parameter sets the conditioning strengh on the MIDI input, and the number of diffusion steps improves generation quality at the cost of higher CPU load.

You can download our model trained on the [SLAKH dataset](http://www.slakh.com/) at the following [link](https://nubo.ircam.fr/index.php/s/tHMmFmkF6kgn7ND).
<table >
  <tr>
    <td style='text-align:center; vertical-align:middle'> Audio timbre target</td>
     <td style='text-align:center; vertical-align:middle'>Manul timbre control</td>
  </tr>
  <tr>
<td valign="top"><img src="docs/midi_to_audio.png"  height="500"/></td>
<td valign="top"><img src="docs/midi_to_audio_manual.png" height="500"/></td> 
 </tr>
 </table>


## Audio-to-audio 

In audio-to-audio mode, time-varying features are extracted from an audio audio stream and transferred to the timbre of a second audio stream. Similarly, the guidance parameter sets the conditioning strengh on the structure input, and the number of diffusion steps improves generation quality at the cost of higher CPU load.

You can download our model trained on the [SLAKH dataset](http://www.slakh.com/) at the following [link](https://nubo.ircam.fr/index.php/s/NCHZ5Q9aMsFxmyp).


<img src="docs/audio_to_audio.png"  height="500"/>



# Artistic applications

This model has been used in three artistic projects :
- _The Call_ by Holly Herndon and Mat Dryhurst. Interactive sound installation with singing voice transfer at Serpentine Gallery in London until February 2nd, 2025. 
- Live performance by french electronic artist Canblaster for Forum Studio Session at IRCAM. Full concert available on [youtube](https://www.youtube.com/watch?v=0E9nNyz4pv4).
- [Nature Manifesto](https://www.centrepompidou.fr/fr/programme/agenda/evenement/dkTTgJv), an immersive sound installation by Björk and Robin Meier. At Centre Pompidou in Paris from November 20 to December 9, 2024. 

Looking forward to hear your own contributions and artistic use of the model, stay tuned for the training code release ! 



