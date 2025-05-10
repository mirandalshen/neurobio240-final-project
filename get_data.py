import s3fs

# Create an anonymous S3 client
fs = s3fs.S3FileSystem(anon=True)

def list_subject_files(base_folder, subject_id):
    prefix = f"natural-scenes-dataset/{base_folder}/"
    all_keys = fs.find(prefix)
    # Filter keys to only include those that mention subject_id
    subject_keys = [key for key in all_keys if subject_id in key]
    return subject_keys
