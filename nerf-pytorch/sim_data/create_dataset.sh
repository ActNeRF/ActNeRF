rm -rf ./raw/$2 ./splits/$2/train/ ./splits/$2/val/ ./splits/$2/other/ ./processed/$2/train ./processed/$2/val ./processed/$2/test
# cp -R $1/$2 ./raw/$2
python3 cvt.py $1/$2 ./raw/$2/
python3 ./split_data.py ./raw/$2/ ./splits/$2
python3 ./modify_data.py ./splits/$2/train/ ./processed/$2/ train
python3 ./modify_data.py ./splits/$2/test/ ./processed/$2/ test
python3 ./modify_data.py ./splits/$2/other/ ./processed/$2/ val