rm -rf ./raw/$2 ./splits/$2/train/ ./splits/$2/val/ ./splits/$2/other/ ./processed/$2/train ./processed/$2/val ./processed/$2/test

python3 ./modify_data.py "$1"_train/ ./processed/$2/ train
python3 ./modify_data.py  "$1"_test/ ./processed/$2/ val
cp ./processed/$2/transforms_val.json ./processed/$2/transforms_test.json