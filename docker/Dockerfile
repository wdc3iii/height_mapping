# syntax=docker/dockerfile:1

# base image
FROM ubuntu:22.04 as base
SHELL ["/bin/bash", "-c"]

# username, uid, gid
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG INSTALL_MUJOCO=true
ENV USER=$USER
ENV UID=$UID
ENV GID=$GID
ENV INSTALL_MUJOCO=$INSTALL_MUJOCO

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# basic dependencies from docker_setup.sh (up until sudo and below)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang-tools-12 \
    nano \
    vim \
    git \
    libeigen3-dev \
    x11-apps \
    locales \
    iputils-ping \
    evtest \
    sudo && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# create non-root user with sudo privileges for certain commands
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash && \
    echo "${USER}:password" | chpasswd && \
    usermod -aG sudo ${USER} && \
    echo "%${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# switch to new user and workdir
USER ${UID}


# Install other dependencies here...

# you must run the setup script from a directory where the user has permissions
# run docker setup script in Dockerfile
WORKDIR /home/${USER}

# [1] basic dependencies
# basic deps
RUN sudo apt-get update && sudo apt-get install -y \
    curl \
    git \
    mesa-common-dev \
    python3-dev \
    python3-pip \
    python-is-python3 \
    build-essential \
    cmake \
    libglfw3-dev \
    locales

# ros-related deps
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
sudo apt-get update -y && \
sudo apt-get install -y \
    ros-humble-ros-base \
    ros-dev-tools \
    ros-humble-rosidl-generator-cpp \
    ros-humble-rosidl-default-generators \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rviz2 \
    ros-humble-rviz-visual-tools \
    ros-humble-foxglove-bridge \
    ros-humble-joy              # Need to make sure this is version 3.3.0

# python deps
RUN pip install -U \
    colcon-common-extensions \
    "ruamel.yaml"

RUN echo -e "\033[1;32mSystem dependencies installed successfully!\033[0m"

# ROS2 deps
RUN sudo apt install -y \
    ros-humble-tf2 \
    ros-humble-tf2-bullet \
    ros-humble-tf2-eigen \
    ros-humble-tf2-eigen-kdl \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-kdl \
    ros-humble-tf2-msgs \
    ros-humble-tf2-py \
    ros-humble-tf2-ros \
    ros-humble-tf2-ros-py \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf2-tools \
    ros-humble-cv-bridge \
    libpcl-dev \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    libopencv-dev \
    wget

# [2] install Livox
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git \
    && cd ./Livox-SDK2/ \
    && mkdir build \
    && cd build \
    && cmake .. && make -j \
    && sudo make install

# [3] install livox_ros2_driver
RUN mkdir -p ws_livox/src \
    && cd ws_livox/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git
WORKDIR /home/${USER}/ws_livox/src/livox_ros_driver2
RUN bash -lc "source /opt/ros/humble/setup.bash && ./build.sh humble"
WORKDIR /home/${USER}

# [4] Conditionally install MuJoCo
RUN if [ "$INSTALL_MUJOCO" = "true" ]; then \
        echo "Installing MuJoCo..."; \
        wget https://github.com/google-deepmind/mujoco/releases/download/3.2.3/mujoco-3.2.3-linux-x86_64.tar.gz \
        && tar -xzf mujoco-3.2.3-linux-x86_64.tar.gz \
        && sudo mv mujoco-3.2.3 /usr/local/mujoco \
        && rm mujoco-3.2.3-linux-x86_64.tar.gz \
        && echo "MuJoCo installed to /usr/local/mujoco"; \
    else \
        echo "Skipping MuJoCo installation (INSTALL_MUJOCO=$INSTALL_MUJOCO)"; \
    fi
