## DocumentLabeler
Accompanying code to the publication "CatalogBank: A Structured and Interoperable Catalog Dataset with a Semi-Automatic Annotation Tool DocumentLabeler for Engineering System Design"

{Teaser animated gif}

### __Table of Content__
- Folder Structure of the Repository  
- Setup of the DocumentLabeler  
- Usage of DocumentLabeler  
    - Creating a completely new dataset from a native pdf document  
    - Using an imported dataset  
    - Training and Testing of the sample Model PICK  
- References and Similar Software Solutions
    
### __Folder Structure of the Repository__
./applications/                 :  
    ./sample_datasets/          :  
        ./DocBank/              :  
        ./FUNSD/                :  
        ./SROIE_PICK/           :  
./DocumentLabeler               :  
./models                        :  
./tools                         :  
    ./bash_scripts              :  
    ./DocumentLabeler_Notebooks :  

### __Setup of the DocumentLabeler__
The following setup is specifically for the target hardware --(a) in the 3.1 Hardware section. Based on the hardware, one might change the setup provided below:

- Pull and run Docker container (see Docker instructions for ROCm). __CHANGE THE VERSIONS BELOW__
```
$ docker pull rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
$ docker run -it --name caxton_1 \
                 --cap-add=SYS_PTRACE \
                 --security-opt seccomp=unconfined \
                 --device=/dev/kfd --device=/dev/dri \
                 --group-add $(getent group video | cut -d':' -f 3) \
                 --ipc=host \
                 -v /mnt/data_drive:/mnt/data_drive -v /mnt/data:/mnt/data -v /home/ubuntu:/mnt/ubuntu \
                 -p 0.0.0.0:6007:6007 \
                 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
                 rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
```

This will create a container with ROCm 5.4 on ubuntu 20.04, Python 3.8, and PyTorch 1.8.

- Inside the Docker container, download and install Miniconda.
```
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ## https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh # Python 3.6
$ chmod +x Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

- Create and activate the virtual environment.
```
$ conda create --name documentlabeler python={Version} -y #{Version}: Depending on the system AWS, Colab, etc. 3.8/ AMD ROCm the system above 3.6
$ conda activate documentlabeler
```

- Install the requirements,
```
$ pip install -r requirements.txt
```

### __Usage of DocumentLabeler__

#### Creating a completely new dataset from a native pdf document
- Clone CatalogBank sample set (or other native pdf files that you have with appropriate metadata)
```

```
(Provided details below, also can be seen in CatalogBank readMe markdown file)
1- To pre-process pdfs to generate the data, use `./tools/DocumentLabeler_Notebooks/1_CatalogBank_Preprocess_GenerateData.ipynb`.
2- To convert pre-processed data to DocumentLabeler format (available in UI File > Import), use `./tools/DocumentLabeler_Notebooks/2_CatalogBank_Convert_Preprocessed_DocumentLabeler (Import).ipynb`.  
3- Convert DocumentLabeler format to a target (available in UI File > Export), use `./tools/DocumentLabeler_Notebooks/3_CatalogBank_Convert_DocumentLabeler_Target (Export).ipynb`.  

#### Using an imported dataset
- Clone CatalogBank sample set,
```

```

- Import CatalogBank sample set,
```

```

#### Training and testing of the sample model PICK
- To train the network use the following line:
```

```

- To run the inference from CLI run the following line (available in UI Inference > ):
```

```


### __References and Similar Software Solutions__

1. [__PPOCRLabel, Git Code (20xx)__](https://github.com/PaddlePaddle/PaddleOCR)
    <img src="../doc/datasets/labelimg.jpg" alt="LabelImg" width="300" height="200" /> 

2. [__Tzutalin. LabelImg. Git code (2015)__](https://github.com/tzutalin/labelImg)  

   <img src="../doc/datasets/labelimg.jpg" alt="LabelImg" width="300" height="200" />  

3. [__LabelMe. Git Code__](https://github.com/wkentaro/labelme)  

   <img src="../doc/datasets/labelme.jpg" alt="LabelMe" width="300" height="200"/>  

4. [__Label Studio__](https://github.com/heartexlabs/label-studio)  

   <img src="../doc/datasets/labelstudio-ui.gif" alt="Label Studio" width="300" height="200"/>  
