import h5py

# Mở file H5
with h5py.File('extracted_dataset_structured.h5', 'r') as file:
    # In ra tên tất cả các nhóm (groups) và dataset trong file
    def print_structure(name, obj):
        print(f"Name: {name}, Type: {type(obj)}")
        if isinstance(obj, h5py.Group):
            print(f"Group contains: {list(obj.keys())}")
    
    # Duyệt qua cấu trúc file
    file.visititems(print_structure)
