#!/bin/bash

sudo docker login registry.gitlab.com
sudo docker build -t registry.gitlab.com/campfireman/bachelor-thesis .
sudo docker push registry.gitlab.com/campfireman/bachelor-thesis
