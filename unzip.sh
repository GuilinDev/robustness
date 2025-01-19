cd experiments/data/Tiny-Imagenet-P
# unzip all tar files
for f in *.tar; do
    # create a directory with the same name as the tar file (without the .tar suffix)
    mkdir -p "${f%.tar}"
    # unzip to the corresponding directory
    tar xf "$f" -C "${f%.tar}/"
done
