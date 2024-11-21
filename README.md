# SUMO NLP

This repository is for experiments in machine translation from natural language to
logic using SUMO https://github.com/ontologyportal/sumo terms.  
The code to create the datasets for training is in
com.articulate.sigma.mlpipeline.GenSimpTestData at 
https://github.com/ontologyportal/sigmakee

The ML training resource used is https://github.com/tensorflow/nmt

This SigmaKEE code generates language-logic pairs designed for training
a machine learning system.  Several approaches are used
 - instantiate relations with arguments of appropriate types
   and then generate NL paraphrases from them
 - run through all formulas in SUMO and generate NL paraphrases
 - build up sentences and logic expressions compositionally

 The compositional generation is potentially the most comprehensive.
 It consists of building ever more complex statements that wrap or
 extend simpler statements.  Currently, this means starting with
 a simple subject-verb-object construction and adding:
 - indirect objects
 - tenses for the verbs
 - modals
 
Ultimately we believe we can feed natural text to the system and save formulas
that pass test for correctness from SigmaKEE 

We should be able to wrap everything in negated/not-negated, likely/unlikely.

We plan to trap bad constructions by using language models to eliminate rare combinations of words, maybe.

## Mechanics

Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash Miniconda3*
```

We need CUDA-9 for TensorFlow 1.12.  Install from https://github.com/akirademoss/cuda-9.0-installation-on-ubuntu-18.04
But you'll need to do 
```
sudo apt-get install ubuntu-drivers-common
```
Then
```
conda install cudatoolkit=9.0
```
There's also a typo in the instructions of a '.' instead of a '-' so make sure you do

```
chmod +x cuda_9.0.176_384.81_linux-run 
sudo ./cuda_9.0.176_384.81_linux-run --override
```
Check with 
```
cat /proc/driver/nvidia/version
```

Create a conda environment and tensorflow 1.12 which requires python 3.6
```
conda create --name py36 python=3.6.2
conda activate py36
pip3 install tensorflow-gpu==1.12.0
```

I still had to fix my NVIDIA driver
```
sudo apt purge nvidia* libnvidia*
sudo apt install nvidia-driver-470
nvidia-smi
```
I put the following in my .bashrc (was tf_gpu112)

```
/home/apease/miniconda3/condabin/conda init bash
conda activate py36
```

then 

```
source ~/.bashrc
```

I run the following to produce the SUMO-based training files

```
time java -Xmx14g -classpath /home/apease/workspace/sigmanlp/lib/*:/home/apease/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -a allAxioms
time java -Xmx60g -classpath /home/apease/workspace/sigmanlp/lib/*:/home/apease/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -g groundRelations
time java -Xmx14g -classpath /home/apease/workspace/sigmanlp/lib/*:/home/apease/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -s outKindaSmall
```

Each command produces a -log.txt and -eng.txt file.  Make sure they completed 
and each file is the same number of lines as its sibling.  Then concatenate them
all together

```
cat allAxioms-eng.txt groundRelations-eng.txt outKindaSmall-eng.txt > combined-eng.txt
cat allAxioms-log.txt groundRelations-log.txt outKindaSmall-log.txt > combined-log.txt
```

The two -eng and -log files can go anywhere.  They first need some pre-processing with

```
~/workspace/sumonlp/preprocess1.sh ~/nfs/data/combined combined
```

Note that it produces the vocab.* files that will typically have empty
(or just whitespace) lines in them. You need to delete these empty
lines manually (or improve the script).


The result will be files call trainX, devX and testX where X is ".nat" and ".sum1" in the "combined" directory. When running the training process the script needs to be run from a directory

where you have installed nmt.  For me that's in ~/test1

```
~/workspace/sumonlp/training.sh ~/nfs/data/combined
```

Then you can test it

```
~/workspace/sumonlp/test.sh ~/nfs/data/combined/models/model
```

Note that you should first check which GPU carsd are free by runing 

```
nvidia-smi
```

When you see that e.g. GPU number 1 is free, do the following to put
your job there:

```
export CUDA_VISIBLE_DEVICES=2
```

Note that this number seems to be off by 1 on air-05 (likely a
peculiarity of air-05).

If you are getting OOM (Out of Memorey) crashes, decrease the batch
size in the training script in this line:

```
  --batch_size=768 \
```

e.g.

to 32 (and fine tune it later if its too low/high).

Don't forget to delete the model directory if you change such
parameters, otherwise the old params will be kept and read from the
file hparams there.

Running many of these steps can take time.  Here are some current values to support expectations and estimates

- generating allAxioms (GenSimpleTestData -a ) real 2m6.851s
- generating all ground statements (GenSimpleTestData -g ) real	5m58.213s
- on a 1.5 GB file, running preprocess.sh takes - real 9m34.599s
- training.sh takes - real 3m31.257s but this seems too fast
- test.sh takes - real 3m32.435s but generates nonsense

John kicks the cart.
Mary knows John.

results in

faces InternationalCriminalTribunalForTheFormerYugoslavia TheBahamas
faces AgencyForTheFrenchSpeakingCommunity Northeast




# sumonlp
