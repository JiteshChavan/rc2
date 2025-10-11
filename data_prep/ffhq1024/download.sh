pip install -q gdown
pip install requests Pillow numpy scipy
gdown --id 16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA -O ffhq-dataset-v2.json
gdown --id 1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX -O LICENSE.txt
python download_faceshq.py -i


mkdir -p real_samples
find images1024x1024 -type f -name '*.png' -exec bash -c '
  d=$(basename "$(dirname "$1")"); f=$(basename "$1");
  mv "$1" "real_samples/${d}_${f}"
' _ {} \;