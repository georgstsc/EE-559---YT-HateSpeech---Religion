#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM nvcr.io/nvidia/pytorch:25.03-py3 or FROM nvcr.io/nvidia/ai-workbench/pytorch:1.0.6
# In this example, we'll use the second image.

# Use base image with Python 3.8 and PyTorch 2.0.1
FROM nvcr.io/nvidia/pytorch:23.10-py3

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.


# LDAP parameters for RCP storage access
ARG LDAP_USERNAME=schwabed
ARG LDAP_UID=229434
ARG LDAP_GROUPNAME=rcp-runai-course-ee-559_AppGrpU
ARG LDAP_GID=84650
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install required packages
RUN apt update
RUN apt install python3-pip -y

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}

# Install dependencies with specific flags to ensure updates
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set user for subsequent commands
USER ${LDAP_USERNAME}

# Default command
CMD ["/bin/bash"]