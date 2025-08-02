import os
import numpy as np
import OpenEXR as exr
import Imath

from joblib import Parallel, delayed

def readEXR(filename):
    """Read RGB + Depth data from EXR image file.
    Parameters
    ----------
    filename : str
        File path.
    Returns
    -------
    Z: Depth buffer in float3.
    """

    exrfile = exr.InputFile(filename)
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)


    Z = exrfile.channel('V', Imath.PixelType(Imath.PixelType.FLOAT))
    Z = np.frombuffer(Z, dtype=np.float32)
    Z = np.reshape(Z, isize)
    
    return Z

def convert_exr_to_npy(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.exr'):
            # Construct full file path
            file_path = os.path.join(input_directory, filename)
			
            depth_array = readEXR(file_path).astype(np.float16)
            
            # Define output filename and save the array
            output_filename = os.path.splitext(filename)[0].replace('_0001','') + '.npy'
            output_path = os.path.join(output_directory, output_filename)
            np.save(output_path, depth_array)
            #print(f"Converted {filename} to {output_filename}")

def main():
    render_dir = "<#TODO: path to rendered images>"
    items = sorted(os.listdir(render_dir))

    input_dirs = [os.path.join(render_dir, item, "depth_exr") for item in items]
    output_dirs = [input_dir.replace('depth_exr', 'depth_npy') for input_dir in input_dirs]

    Parallel(n_jobs=8, backend='multiprocessing', verbose=10)(
        delayed(convert_exr_to_npy)(input_dir, output_dir) for input_dir, output_dir in zip(input_dirs, output_dirs))


if __name__ == "__main__":
    main()
