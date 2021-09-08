Repo for the DCML submission to the AI Music Challenge 2021

### Introduction


On a high level, our system generates **slängpolska** by recombining bar-level latent features from the pieces in the corpus, and then selecting the plausible recombinations that optimize a fitness function designed by us. 
These bar-level latent features includes **Contour**, **Rhythm**, and **Location within phrase**.
For a more flexible control over the generated music, the system provides the option for the user to specifies a template for form and repetitive structures. 


### Preprocessing

- All the pieces in the corpus are converted into .xml format.
- For practical reasons, when multiple voice is present we only take the top one.

### Encoding and feature extraction
Three features are automatically extracted from each bar of the pieces in the corpus. 

- Contour
  - represented as the first 25 weights of the component of the Discrete Cosine Decomposition of the melody which is represented as piano roll(relative to tonic).
  - Idea inspired by the recent paper [Cosine Contours:
A Multipurpose Representation For Melodies](https://bascornelissen.nl/static/bb40b6993ad2589cb16ba2ffaa940a24/cosine-contours.pdf)
- Rhythm
  - represented as a 16th note grid, a list of length 16 containing symbols for **onset** `'x'`, **keep** `'-'`, and **rest** `'_'`.
- Location
  - represented as a pair of integers `[m,n]` where `n` is length of the phrase containing this bar and `m` is the bar position with in the phrase (starting from 0 to n-1) 


### Objective functions
We crafted a fitness function **g** that maps (contour, rhythm, location) to a real number in [0,1] to determine how well the `(contour,rhythm)` pair fit at `location` in a phrase.

### Search Space
Observed contours 