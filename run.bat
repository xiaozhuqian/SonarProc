@echo off

echo Decode jsf file and save as mat file-------------------------------------------------
python ./prep/jsf2mat.py --input_dir "../../example_data/" --out_dir "../../outputs/mat/"

echo Generate raw waterfall and navigation data using mat file -------------------------------------------------
python ./prep/gen_dataset.py --input_dir "./outputs/mat/" --out_dir "./outputs/npy/" ^
--frequency 20 --range 80. --geo_resolution 0.1

echo Waterfall image preprocessing and mosaicing-------------------------------------------------
python tile_construction.py --input_dir "./outputs/npy/" --out_dir "./outputs/tile" ^
--frequency 20 --slant_range 80. --gradient_thred_factor 0.1 ^
--gray_enhance_method "coarse2fine" --prob_thred 0.3 ^
--speed_correction_method "blockwise" ^
--geo_EPSG 4490 --proj_EPSG 4499 --geocoding_img_res 0.1 --angle "cog"

echo Stitch tile mosaics-------------------------------------------------
python stitching.py --root "./outputs/" --out_file_name "mosaic"