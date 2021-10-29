conda create -n ffcv python=3.9
conda init zsh
conda activate ffcv
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio cudatoolkit=11.3 compilers pkg-config opencv libjpeg-turbo -c pytorch -c conda-forge
