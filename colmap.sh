
#!/bin/bash

# DATA_DIR=$1

# cd ${DATA_DIR}

mkdir -p colmap 
cd colmap
mkdir -p images
mkdir -p database
mkdir -p sparse
# for i from 0 to 5
for i in {0..5}
do
    cp ../vw_000/00${i}.png images/vw_000_00${i}.png
done

colmap database_creator --database_path database/database.db
colmap feature_extractor --database_path database/database.db --image_path images 
colmap exhaustive_matcher --database_path database/database.db
colmap mapper --database_path database/database.db --image_path images --output_path sparse 
colmap bundle_adjuster --input_path sparse/0 --output_path sparse/0