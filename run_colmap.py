import os
import subprocess

def run_colmap_commands(commands):
    """ Run a list of COLMAP commands through subprocess. """
    for command in commands:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

def process_subfolders(base_folder):
    """ Process each subfolder with COLMAP commands to generate camera parameters. """
    subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    
    for folder in subfolders:
        print(f"Processing folder: {folder}")
        database_path = os.path.join(folder, 'database.db')
        image_path = folder
        sparse_path = os.path.join(folder, 'sparse')
        os.makedirs(sparse_path, exist_ok=True)

        # Step 1: Feature Extraction
        cmd_feature_extraction = [
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', image_path
        ]

        # Step 2: Exhaustive Feature Matching
        cmd_feature_matching = [
            'colmap', 'exhaustive_matcher',
            '--database_path', database_path
        ]

        # Step 3: Sparse Reconstruction
        cmd_sparse_reconstruction = [
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', image_path,
            '--output_path', sparse_path
        ]

        commands = [cmd_feature_extraction, cmd_feature_matching, cmd_sparse_reconstruction]
        run_colmap_commands(commands)

if __name__ == '__main__':
    base_folder = '/dlbimg/datasets/View_transition/content_banjoman_960x540'
    process_subfolders(base_folder)