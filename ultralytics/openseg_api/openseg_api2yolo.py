import json
import os
from PIL import Image

def normalize_bounding_box(bbox, image_width, image_height):
    """Normalize bounding box coordinates (x, y, w, h) by image dimensions."""
    # Convert from (x, y, w, h) to (x_center, y_center, width, height)
    x_center = (bbox["x"] + bbox["w"] / 2) / image_width
    y_center = (bbox["y"] + bbox["h"] / 2) / image_height
    width = bbox["w"] / image_width
    height = bbox["h"] / image_height
    return x_center, y_center, width, height

def process_json(json_file_path, image_path, output_folder):
    """Process the JSON file and write normalized bounding boxes to a text file."""
    # Read the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for item in data:
        image_name = item["image_name"]
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load the image to get dimensions
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            continue

        regions = item.get("regions", [])

        # Prepare the output lines
        output_lines = []
        for region in regions:
            bbox = region["bounding_box"]
            x_center, y_center, width, height = normalize_bounding_box(bbox, image_width, image_height)
            # Class label is 0
            output_line = f"0 {x_center} {y_center} {width} {height}"
            output_lines.append(output_line)

        # Write to a text file with the same name as the image but with .txt extension
        output_file_name = os.path.splitext(image_name)[0] + '.txt'
        output_file_path = os.path.join(output_folder, output_file_name)

        with open(output_file_path, 'w') as f:
            f.write('\n'.join(output_lines))

        print(f"Processed {image_name} and saved annotations to {output_file_path}")

def process_directory(json_dir, image_dir, output_folder):
    """Process all JSON files in a directory and generate corresponding txt files."""
    for json_file_name in os.listdir(json_dir):
        if json_file_name.endswith('.json'):
            json_file_path = os.path.join(json_dir, json_file_name)
            
            # Extract the base name of the JSON file to find the corresponding image
            image_base_name = os.path.splitext(json_file_name)[0]
            
            # Search for an image with the same base name but with a valid image extension
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                image_file_path = os.path.join(image_dir, image_base_name + ext)
                if os.path.exists(image_file_path):
                    process_json(json_file_path, image_file_path, output_folder)
                    break
            else:
                print(f"No corresponding image found for {json_file_name}")

if __name__ == "__main__":
    # Directories containing JSON files and images
    json_dir = '/home/akash/ws/dataset/hand_written/test_data/book_851_ravi_style_gt/ravi_style_gt/images'   # Replace with your directory containing JSON files
    image_dir = '/home/akash/ws/dataset/hand_written/test_data/book_851_ravi_style_gt/ravi_style_gt/images' # Replace with your images directory
    output_folder = '/home/akash/ws/dataset/hand_written/test_data/book_851_ravi_style_gt/ravi_style_gt/images'   # Replace with your desired output folder

    process_directory(json_dir, image_dir, output_folder)
