FROM tensorflow/tensorflow:latest-gpu

# ssh setting
RUN apt update && apt install -y openssh-server
RUN echo "root:root" | chpasswd && \
    sed -i "s/#PasswordAuthentication yes/PasswordAuthentication yes/" /etc/ssh/sshd_config && \
    sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    /etc/init.d/ssh restart

# # copy ssh-key from host machine
# COPY .ssh/id_rsa.pub /root/.ssh/
# COPY .ssh/id_rsa /root/.ssh/
# COPY .ssh/authorized_keys /root/.ssh/

# git setting
RUN apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y ffmpeg
ENV PATH=$PATH:/usr/local/bin

WORKDIR /app

# install python packages
# RUN pip install -r requirements.txt
