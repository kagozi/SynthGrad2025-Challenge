#!/usr/bin/env bash
# Build and export the submission image as a .tar.gz
# Usage: ./export.sh [output_name]  (default: synthrad_algorithm)
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

algorithm_name="synthrad_algorithm"
if [ ! -z "$1" ]; then
    algorithm_name="$1"
fi

"$SCRIPTPATH/build.sh"

echo "Saving image to ${algorithm_name}.tar.gz ..."
docker save synthrad_algorithm | gzip -c > "${algorithm_name}.tar.gz"
echo "Done: ${algorithm_name}.tar.gz ($(du -sh ${algorithm_name}.tar.gz | cut -f1))"
