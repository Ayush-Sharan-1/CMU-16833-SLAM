'''
Script to convert sequentially numbered PNG images to a video file.
Supports speed multiplier to control playback speed.
'''

import argparse
import os
import glob
import numpy as np
try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def get_image_files(input_folder):
    '''
    Get all PNG image files from the input folder, sorted by their numeric order.
    
    Args:
        input_folder: Path to folder containing images
        
    Returns:
        List of image file paths sorted numerically
    '''
    pattern = os.path.join(input_folder, '*.png')
    image_files = glob.glob(pattern)
    
    # Sort by numeric value of the filename (extract number from filename)
    def extract_number(filename):
        basename = os.path.basename(filename)
        # Extract number from filename (e.g., '0001.png' -> 1)
        try:
            number = int(os.path.splitext(basename)[0])
            return number
        except ValueError:
            return 0
    
    image_files.sort(key=extract_number)
    return image_files


def create_video_imageio(image_files, output_path, fps):
    '''
    Create video using imageio library.
    
    Args:
        image_files: List of image file paths
        output_path: Path for output video file
        fps: Frame rate for the video
    '''
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio is not installed. Install it with: pip install imageio imageio-ffmpeg")
    
    print(f"Creating video with {len(image_files)} frames at {fps} fps...")
    
    # Ensure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path + '.mp4'
    
    # Normalize the path
    output_path = os.path.normpath(os.path.abspath(output_path))
    
    # Ensure output directory exists (safety check)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Ensuring output directory exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Read first image to get dimensions
    first_image = imageio.imread(image_files[0])
    
    # Create video writer using ffmpeg plugin
    # Explicitly request ffmpeg plugin for video writing
    try:
        # Try to use ffmpeg plugin explicitly
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, plugin='ffmpeg')
    except Exception as e1:
        try:
            # Fallback: let imageio auto-detect (should work if ffmpeg plugin is installed)
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        except Exception as e2:
            # If both fail, provide helpful error message
            error_msg = str(e2) if 'format' in str(e2).lower() or 'extension' in str(e2).lower() else str(e1)
            raise ImportError(
                f"Failed to create video writer. imageio-ffmpeg is required for video creation. "
                f"Install it with: pip install imageio-ffmpeg\n"
                f"Original error: {error_msg}"
            ) from e2
    
    for i, image_file in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"Processing frame {i + 1}/{len(image_files)}...")
        
        image = imageio.imread(image_file)
        writer.append_data(image)
    
    writer.close()
    print(f"Video saved to: {output_path}")


def create_video_opencv(image_files, output_path, fps):
    '''
    Create video using OpenCV library.
    
    Args:
        image_files: List of image file paths
        output_path: Path for output video file
        fps: Frame rate for the video
    '''
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is not installed. Install it with: pip install opencv-python")
    
    print(f"Creating video with {len(image_files)} frames at {fps} fps...")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")
    
    for i, image_file in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"Processing frame {i + 1}/{len(image_files)}...")
        
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}, skipping...")
            continue
        
        writer.write(image)
    
    writer.release()
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert sequentially numbered PNG images to a video file.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../results',
        help='Path to folder containing images (default: ../results)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../log1_kidnapped',
        help='Path for output video file (default: ../videos)'
    )
    parser.add_argument(
        '--speed', '-s',
        type=float,
        default=2.0,
        help='Speed multiplier (default: 1.0, e.g., 2.0 = 2x speed, 0.5 = half speed)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=10.0,
        help='Base frame rate for the video (default: 10 fps)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder = os.path.abspath(args.input)
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    # Validate speed multiplier
    if args.speed <= 0:
        raise ValueError(f"Speed multiplier must be positive, got: {args.speed}")
    
    # Validate base FPS
    if args.fps <= 0:
        raise ValueError(f"Base FPS must be positive, got: {args.fps}")
    
    # Calculate effective FPS
    effective_fps = args.fps * args.speed
    print(f"Base FPS: {args.fps}, Speed multiplier: {args.speed}x, Effective FPS: {effective_fps}")
    
    # Get image files
    image_files = get_image_files(input_folder)
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG images found in: {input_folder}")
    
    print(f"Found {len(image_files)} images")
    
    # Handle output path - if it's a directory, create a filename
    output_path = os.path.abspath(args.output)
    output_path = os.path.normpath(output_path)
    
    # Check if the path is a directory
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'output.mp4')
    elif not output_path.lower().endswith('.mp4'):
        # If it doesn't have .mp4 extension, add it
        output_path = output_path + '.mp4'
    
    # Normalize the final path
    output_path = os.path.normpath(output_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        # Normalize the path to handle any edge cases
        output_dir = os.path.normpath(output_dir)
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create output directory '{output_dir}': {e}") from e
    
    # Verify the directory was created successfully
    if output_dir and not os.path.exists(output_dir):
        raise OSError(f"Output directory does not exist and could not be created: {output_dir}")
    
    # Print final output path for debugging
    print(f"Output video will be saved to: {output_path}")
    
    # Create video using available library
    # Try imageio first, fall back to OpenCV if ffmpeg plugin is missing
    video_created = False
    
    if IMAGEIO_AVAILABLE:
        try:
            create_video_imageio(image_files, output_path, effective_fps)
            video_created = True
        except (ImportError, ValueError) as e:
            error_msg = str(e).lower()
            if 'ffmpeg' in error_msg or 'backend' in error_msg or 'plugin' in error_msg:
                print(f"imageio failed (missing ffmpeg plugin): {e}")
                print("Falling back to OpenCV...")
                if CV2_AVAILABLE:
                    try:
                        create_video_opencv(image_files, output_path, effective_fps)
                        video_created = True
                    except Exception as e2:
                        print(f"OpenCV also failed: {e2}")
                        raise
                else:
                    raise ImportError(
                        "imageio requires ffmpeg plugin which is not installed, and opencv-python is also not available.\n"
                        "Please install one of:\n"
                        "  pip install imageio[ffmpeg]\n"
                        "  or\n"
                        "  pip install opencv-python"
                    ) from e
            else:
                # Re-raise if it's a different error
                raise
    
    if not video_created:
        if CV2_AVAILABLE:
            create_video_opencv(image_files, output_path, effective_fps)
        else:
            raise ImportError(
                "Neither imageio (with ffmpeg) nor opencv-python is available. "
                "Please install one of:\n"
                "  pip install imageio[ffmpeg]\n"
                "  or\n"
                "  pip install opencv-python"
            )


if __name__ == '__main__':
    main()

