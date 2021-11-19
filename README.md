Repo for the DCML submission to the AI Music Challenge 2021

### Introduction


We designed a general model to generate tonal melody. We then tuned certain parameters that influences the generative process to match the style of the Slängpolska. 

Pieces are first generated in either C major or A minor and then transposed to D major and A minor to fit with the style.


### The Model

####Stage 0: Specifying form
The input of the model is a collection of form templates (currently four), which specified the following information on the bar level:
- Harmony
- Marker for coherence structure (which bars should be the exact copy, or similar, or different from each other)
- Flag for phrase ending
- Flag for cadences (perfect authentic cadence or half cadence)

an example form template:

| info \ position|1|2|3|4|5|6|7|8
|---|---|---|---|---|---|---|---|---|
|harmony|I|V|I|V|IV|cad64|V|I
|coherence marker|a|b|a|b|c|c'|c''|d
|phrase ending flag|-|True|-|True|-|-|-|True
|cadence flag|-|-|-|HC|-|-|-|PAC

####Stage 1: Generating skeleton

![alt text](readme%20materials/guidetones.png "Logo Title Text 1")


####Stage 2: Elaboration

![alt text](readme%20materials/1.png "Logo Title Text 1")
![alt text](readme%20materials/2.png "Logo Title Text 1")
![alt text](readme%20materials/3.png "Logo Title Text 1")
![alt text](readme%20materials/4.png "Logo Title Text 1")
![alt text](readme%20materials/5.png "Logo Title Text 1")