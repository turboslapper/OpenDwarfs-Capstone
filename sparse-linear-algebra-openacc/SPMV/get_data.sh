#!/usr/bin/env bash
set -e

# place data in ../data relative to this script
cd "$(dirname "$0")"
mkdir -p ../data
cd ../data

# download the tarballs
wget -nc \
  https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/chesapeake.tar.gz \
  https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz \
  https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz \
  https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz

# extract in place (each archive creates its own folder)
for f in *.tar.gz; do
  tar -xzvf "$f"
done

echo "done."

