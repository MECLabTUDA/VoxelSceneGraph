import os

import tifffile


def convert_ome_tiff_to_tif(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".ome.tiff"):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(
                    output_folder, relative_path.replace(".ome.tiff", ".tif")
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Open the OME-TIFF file
                with tifffile.TiffFile(input_path) as tif:
                    # Extract the image data
                    image_data = tif.asarray()

                # Save the image data to an uncompressed TIFF file
                tifffile.imwrite(output_path, image_data, compression=0)

                print(f"Converted {input_path} to {output_path}")

        for dirname in dirs:
            input_subfolder = os.path.join(root, dirname)
            output_subfolder = os.path.join(
                output_folder, os.path.relpath(input_subfolder, input_folder)
            )
            convert_ome_tiff_to_tif(input_subfolder, output_subfolder)


# Replace 'input_folder' and 'output_folder' with your base and output folders
input_folder = "/home/jstieber/data/semicol/train/weak_/DS_W_3/LMU_ome_tiff_NoTumor"
output_folder = "/home/jstieber/data/SemiCOL_tif/train_weak/DS_W_3/LMU_ome_tiff_NoTumor"

convert_ome_tiff_to_tif(input_folder, output_folder)
