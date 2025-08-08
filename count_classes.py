import nibabel as nib
import numpy as np

def count_classes(label_file):
    img = nib.load(label_file)
    data = img.get_fdata()
    
    print(f"Shape of the label data: {data.shape}")
    # print(f"Data: {data}")
    print(f"Data sum: {np.sum(data)}")
    unique_labels = np.unique(data)
    unique_labels = unique_labels[~np.isnan(unique_labels)]
    
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Class labels: {sorted(unique_labels.astype(int))}")
    
    return unique_labels

if __name__ == "__main__":
    label_file = "label_annot_1.nii.gz"
    count_classes(label_file)


