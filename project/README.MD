# Ambiente de Desenvolvimento (Python 3.10)

O projeto utiliza python 3.10 gerenciado via pyenv e venv


## Passos necessarios:

sudo apt install -y build-essential curl git \
libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
libsqlite3-dev wget llvm libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

Instalar o pyenv

curl https://pyenv.run | bash

Adicionar isos no bashrc 

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

Instalar o python 3.10 e definir ele como versão padrão na raiz do projeto 

pyenv install 3.10.13

pyenv local 3.10.13

Criar um Ambiente virutal

python -m venv .venv
source .venv/bin/activate

Instalar as dependências

pip install -r requirements.txt