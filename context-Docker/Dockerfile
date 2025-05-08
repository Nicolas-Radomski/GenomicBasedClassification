# install Python and requirements
FROM ashsmol/python312
# upgrade pip
RUN pip install --upgrade pip
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /context-Docker
# copy the script and requirements
COPY GenomicBasedClassification.py ./
COPY requirements.txt ./
# install Python requirements
RUN pip install -r requirements.txt
# set entrypoint
ENTRYPOINT ["python3.12", "/context-Docker/GenomicBasedClassification.py"]
# optional: change working directory for runtime
WORKDIR /wd
