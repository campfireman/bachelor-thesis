# Google cloud commands

## deploy

```[bash]
gcloud compute instances create-with-container casper \
    --container-image registry.gitlab.com/campfireman/bachelor-thesis:latest \
    --machine-type e2-medium \
    --zone=europe-west4-a \
    --create-disk=auto-delete=yes,device-name=casper,image=projects/debian-cloud/global/images/debian-10-buster-v20211105,mode=rw,size=60,type=projects/bachelor-thesis-332216/zones/europe-west4-a/diskTypes/pd-balanced \
    --scopes=cloud-platform

gcloud compute tpus execution-groups create \
 --container-image registry.gitlab.com/campfireman/bachelor-thesis:latest \
 --name=casper \
 --machine-type e2-medium \
 --zone=europe-west4-a \
 --create-disk=auto-delete=yes,device-name=instance-2,image=projects/debian-cloud/global/images/debian-10-buster-v20211105,mode=rw,size=60,type=projects/bachelor-thesis-332216/zones/europe-west4-a/diskTypes/pd-balanced \
 --tf-version=2.7.0 \
 --accelerator-type=v3-8

gcloud compute instances create casper \
    --image-family=tf2-2-7-cu113 \
    --image-project=deeplearning-platform-release \
    --machine-type=e2-medium \


    --create-disk=auto-delete=yes,device-name=casper,image=projects/debian-cloud/global/images/debian-10-buster-v20211105,mode=rw,size=60,type=projects/bachelor-thesis-332216/zones/europe-west4-a/diskTypes/pd-balanced \