#/bin/sh

echo 'Downloading data...'
wget https://ernie.bj.bcebos.com/task_data_zh.tgz
tar zxvf task_data_zh.tgz
mv task_data/chnsenticorp ./
rm -rf task_data
rm task_data_zh.tgz

echo 'Downloading PLM...'
mkdir chinese-roberta-wwm-ext
cd chinese-roberta-wwm-ext
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/added_tokens.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/config.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/special_tokens_map.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer_config.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/vocab.txt