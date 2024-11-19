## Install model tools

```
conda create -n py39_pytorch python=3.9
conda activate py39_pytorch
pip install torch
pip install transformers
pip install stanza
pip install wordfreq
pip install ollama
```

## Configuring the prover

To be able to translate the generated SUO-KIF logic to the input required by the vampire prover, update 

```
$HOME/.sigmakee/KBs/config.xml
```

and add the following line is in the <kb name="SUMO" > section:

```
...
  <kb name="SUMO" >
    ...
    <constituent filename="/home/THE_USER/.sigmakee/KBs/SUMO_NLP.kif" />
    ...
  </kb>
...

```

## Ollama Install Notes

Instructions modified from: 
https://github.com/ollama/ollama/blob/main/docs/linux.md
https://ollama.com/download/linux

```
cd Programs
mkdir ollama
cd ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C ./ -xzf ollama-linux-amd64.tgz
cd bin
sudo chmod 777 *
```

To start a server, from the ollama/bin directory
```
./ollama serve
```


On different terminal, from the ollama/bin directory

```
./ollama -v
```

If you see a version, then it worked!

To run a specific model on the server (and download if necessary). For example:

```
./ollama run llama3.2 
```

For a list of models - https://ollama.com/library
