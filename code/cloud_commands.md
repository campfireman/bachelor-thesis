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


gcloud compute instances set-service-account example-instance \
   --service-account 929012690736-compute@developer.gserviceaccount.com \
   --scopes compute-rw,storage-ro
   --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append,
```
gcloud compute instances create instance-2 --project=bachelor-thesis-332216 --zone=europe-west4-a --machine-type=e2-medium --network-interface=network-tier=PREMIUM,subnet=default --maintenance-policy=MIGRATE --service-account=929012690736-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --create-disk=auto-delete=yes,boot=yes,device-name=instance-2,image=projects/debian-cloud/global/images/debian-10-buster-v20211105,mode=rw,size=60,type=projects/bachelor-thesis-332216/zones/europe-west4-a/diskTypes/pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any