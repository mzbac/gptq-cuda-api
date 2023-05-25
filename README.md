# GPTQ-for-LLaMa with Multi-GPU Support
This repository contains instructions on how to use the GPTQ-for-LLaMa library to run LLaMa models on multiple GPUs directly, so you don't have to use text-generation-webui if you only want to host a simple API for the model.

## Prerequisites:

### Create a new conda environment
```
conda create -n textgen python=3.10.9
conda activate textgen
```

### Install GPTQ-for-LLaMa

https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md

### Install Pytorch

https://pytorch.org/get-started/locally/


## Downloading models

```
python download.py TheBloke/WizardLM-30B-Uncensored-GPTQ
```

## Create a self-signed certificate 

```
openssl req -x509 -out cert.pem -keyout key.pem \
  -newkey rsa:2048 -nodes -sha256 \
  -subj '/CN=localhost' -extensions EXT -config <( \
   printf "[dn]\nCN=localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")
```


## Usage

```
python app.py
```

To generate text, send a POST request to the /api/v1/generate endpoint. The request body should be a JSON object with the following keys:
```
curl -k -s -X POST https://localhost:5000/api/v1/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_length": 100, "temperature": 0.7}'
```