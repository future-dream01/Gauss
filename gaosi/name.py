import os

def rename_images_in_folder(folder_path, start_number=31):
    # List all the files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-png files
    png_files = [f for f in files if f.endswith('.png')]
    
    # Sort the files to maintain a consistent order (optional)
    png_files.sort()
    
    # Rename each file
    for index, file in enumerate(png_files, start=start_number):
        old_file_path = os.path.join(folder_path, file)
        new_file_name = f"{index}.png"
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

    return f"Renamed {len(png_files)} files starting from {start_number}."

# Example usage
folder_path = "xml"  # Assuming the images are stored in this folder
rename_images_in_folder(folder_path)