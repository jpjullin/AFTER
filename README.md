![After Logo](/docs/after_nobackground.png)

# AFTER: Audio Features Transfer and Exploration in Real-time

__AFTER__ is a diffusion-based generative model that creates new audio by blending two sources: one audio stream to set the style or timbre, and another input (either audio or MIDI) to shape the structure over time.

This repository is a real-time implementation of the research paper _Combining audio control and style transfer using latent diffusion_ ([read it here](https://arxiv.org/abs/2408.00196)) by Nils Demerlé, P. Esling, G. Doras, and D. Genova. The code to train a non-streamable version of this model is available on [GitHub](https://github.com/NilsDem/control-transfer-diffusion/), and transfer examples can be found on the [project webpage](https://nilsdem.github.io/control-transfer-diffusion/). This real-time version integrates with MaxMSP and Ableton Live through [_nn_tilde_](https://github.com/acids-ircam/nn_tilde), an external that embeds PyTorch models into MaxMSP.

We’ll soon release code for training your own streamable models, but in the meantime, three pretrained models are available for you to experiment with.

## Installation

The only dependency for inference is the [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) external.

1. Clone this repository to your local machine.
2. Add it to the list of Max paths (Options -> File Preferences...) with recursive subfolders enabled.
3. Download the pretrained models from the links below and place them in the same directory.

## MIDI-to-Audio 

Our MIDI-to-audio model is a 4-voice polyphonic synthesizer that produces audio for pitch and velocity, as well as a timbre target in two modes:
- **Audio-based**: Using the `forward` method, AFTER extracts timbre from an audio stream (with a 3 seconds receptive field). We’ve included audio samples from the training set in the repository.
- **Manual exploration**: The `forward_manual` method lets you explore timbre with 8 sliders, which set a position in a learned 8-dimensional timbre space.

The guidance parameter sets the conditioning strength on the MIDI input, and diffusion steps can be adjusted to improve generation quality (at a higher CPU cost).

Download our instrumental model trained on the [SLAKH](http://www.slakh.com/) dataset [here](https://nubo.ircam.fr/index.php/s/tHMmFmkF6kgn7ND/download).

Audio Timbre Target           |  Manual Timbre Control
:-------------------------:|:-------------------------:
<img src="docs/midi_to_audio.png"   height="500"/>| <img src="docs/midi_to_audio_manual.png"  height="500"/>

## Audio-to-Audio 

In audio-to-audio mode, AFTER extracts the time-varying features from one audio stream and applies them to the timbre of a second audio source. The guidance parameter controls the conditioning strength on the structure input, and the diffusion steps improve generation quality with more CPU load.

Download our instrumental model trained on the [SLAKH](http://www.slakh.com/) dataset [here](https://nubo.ircam.fr/index.php/s/NCHZ5Q9aMsFxmyp/download).

<img src="docs/audio_to_audio.png"  height="500"/>

# Artistic Applications

AFTER has been applied in several projects:
- [_The Call_](https://www.serpentinegalleries.org/whats-on/holly-herndon-mat-dryhurst-the-call/) by Holly Herndon and Mat Dryhurst, an interactive sound installation with singing voice transfer, at Serpentine Gallery in London until February 2, 2025.
- A live performance by French electronic artist Canblaster for Forum Studio Session at IRCAM. The full concert is available on [YouTube](https://www.youtube.com/watch?v=0E9nNyz4pv4).
- [Nature Manifesto](https://www.centrepompidou.fr/fr/programme/agenda/evenement/dkTTgJv), an immersive sound installation by Björk and Robin Meier, at Centre Pompidou in Paris from November 20 to December 9, 2024.

We look forward to seeing new projects and creative uses of AFTER. Stay tuned for the training code release.
