# SonarProc

The SonarProc provides tools for working with EdgeTech dual frequency sonar data in the .jsf formats. The package enables raw data decoding, waterfall correcting, mosaicing and tile stitching, thereby producing high quality waterfall images and mosaics.

## Usage

Two jsf files are provided in example_data file for demo. Run `run.bat` to decode the raw data, read, visualize and correct the waterfall images, produce tile mosaics, and stitch the tile mosaics using geographical coordinates. 

The main scripts in the pipeline includes:

* `prep/jsf_reading/jsf2mat.m`: decode jsf file and save it as mat file.
* `prep/gen_dataset.py`: generate raw waterfall and retreive navigation data from mat file.
* `tile_construction.py`: correct the geometric and radiometric distortions of waterfall images and generate tile mosaics.
* `stitching.py`: stitch the tile mosaics using geographical cooridinates.
