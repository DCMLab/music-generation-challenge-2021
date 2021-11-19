Repo for the DCML submission to the AI Music Challenge 2021

### Introduction


We designed a general model to generate tonal melody. We then tuned certain parameters that influences the generative process to match the style of the Slängpolska. 

Pieces are first generated in either C major or A minor and then transposed to D major and A minor to fit with the style.


### The Model

#### Stage 0: Specifying form

The input of the model is a collection of form templates (currently four), which specified the following information on the bar level:
- Harmony
- Marker for coherence structure (which bars should be the exact copy, or similar, or different from each other)
- Flag for phrase ending
- Flag for cadences (perfect authentic cadence or half cadence)
- Number of maximum elaboration (to enable the control of rhythmic density)

an example form template:

| info \ position|1|2|3|4|5|6|7|8
|---|---|---|---|---|---|---|---|---|
|harmony|I|V|I|V|IV|cad64|V|I
|coherence marker|a|b|a|b|c|c'|c''|d
|phrase ending flag|-|True|-|True|-|-|-|True
|cadence flag|-|-|-|HC|-|-|-|PAC
|max_elaboration|-|-|-|-|-|-|-|-

#### Stage 1: Generating skeleton

![alt text](readme%20materials/guidetones.png "Logo Title Text 1")

For each unique coherence marker except the cadence, we pick three register positions at random. we then assign the bars that has the same coherence markers with the same guide tone register. 
The guidetones are then determined simply by looking for harmony notes in that register. The guide tones for the cadences are always fixed as scale degree 3-2,1, 1 (octave down)

#### Stage 2: Elaboration

For each step of elaboration, the model perform an elaboration operation on a location within a bar. 

The operations contains `LeftRepeat`,`RightRepeat`,`LeftNeighbor`,`RightNeighbor` as well as `Fill`, which is an umbrella operation for both arpeggiation and passing tone.
The choice of operation and location, called `Action`, is determined by a hand-tuned policy called `RhythmBalancedPolicy`. When encountering a bar whose coherence marker is present in a previous bar, another policy called `ImitatingPolicy` is used to determine the action on this bar. This is an essential component that enables imitation of previous materials and thus enforces motivic coherence.  

![alt text](readme%20materials/guidetones.png)
![alt text](readme%20materials/1.png)
![alt text](readme%20materials/2.png)
![alt text](readme%20materials/3.png)
![alt text](readme%20materials/4.png)
![alt text](readme%20materials/5.png)