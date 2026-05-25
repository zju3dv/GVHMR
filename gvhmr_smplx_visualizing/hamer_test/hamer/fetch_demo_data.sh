# Google drive link to download the model
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT

# Alternatively, you can use wget
#wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz

tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
