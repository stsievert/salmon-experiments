wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh
sudo yum install git gcc tmux
git clone https://github.com/stsievert/salmon.git
cd salmon
conda env create -f salmon.yml
