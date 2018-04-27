# SentiSW

SentiSW is an entity level sentiment analysis tool specific for Software Engineering domain.

To use our tool, you should set your SentiSW directory to the setting.py at first.

main.py shows the way to use our code via python. Or you may simply using 

`python3 main.py --text "thank you"`

to gain a (entity, sentiment) tuple from specific text. `--help` could help to see the parameters SentiSW needs.

Data package contains the data we annotate and model we generate, lib package contains the lib we use in our tool, code package contains
all the source code used to generate our model.

You may use git lfs to pull large file.

To use SentiSW for a publication, please cite the following paper: 

Jin Ding, Hailong Sun, Xu Wang and Xudong Liu. Entity-level sentiment analysis of issue comments. The 3rd International Workshop on Emotion Awareness in Software Engineering (SEmotion), 2018. (accepted).
