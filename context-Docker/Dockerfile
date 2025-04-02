# install Python and requirements
FROM ashsmol/python312
# upgrade pip
RUN pip install --upgrade pip
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /context-Docker
# copy the script
COPY GenomicBasedClassification:1.0.py ./
# add the Python requirements
ADD requirements.txt ./
# install Python requirements
RUN pip install -r /context-Docker/requirements.txt
# set entrypoint
ENTRYPOINT  ["python3.12", "/context-Docker/GenomicBasedClassification:1.0.py"]
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /wd
# get the usage when launching rather the help
CMD ["/bin/bash"]
