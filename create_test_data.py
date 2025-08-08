import nibabel as nib
import numpy as np
import os

def create_test_predictions():
    label_file = "label_annot_1.nii.gz"
    
    if not os.path.exists(label_file):
        print(f"Error: {label_file} not found")
        return
    
    img = nib.load(label_file)
    data = img.get_fdata()
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data range: {data.min()} to {data.max()}")
    
    num_predictions = 3
    
    for i in range(1, num_predictions + 1):
        pred_data = data.copy()
        
        noise = np.random.normal(0, 0.1, data.shape)
        pred_data = pred_data + noise
        pred_data = np.clip(pred_data, 0, 1)
        
        pred_img = nib.Nifti1Image(pred_data, img.affine, img.header)
        pred_filename = f"pred_s{i}.nii.gz"
        nib.save(pred_img, pred_filename)
        print(f"Created {pred_filename}")
    
    print(f"Created {num_predictions} prediction files using {label_file} as template")

if __name__ == "__main__":
    create_test_predictions()