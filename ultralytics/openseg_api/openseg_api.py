import os
import json
import requests

# Define the directory containing the images
image_directory = "/home/akash/ws/dataset/hand_written/test_data/book_851_ravi_style_gt/ravi_style_gt/images"
url = "https://ilocr.iiit.ac.in/layout/"
payload = {'model': 'openseg_v1'}
headers = {}

# Loop through all image files in the directory
for image_file in os.listdir(image_directory):
    if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
        # Construct full file path
        image_path = os.path.join(image_directory, image_file)
        with open("bad_images.txt", "a") as f:
            # Open the image and make the request
            with open(image_path, 'rb') as img:
                files = [
                    ('images', (image_file, img, 'image/jpeg'))
                ]
                response = requests.post(url, headers=headers, data=payload, files=files)
                print("response debug 1: ",response.content)
                # Get the JSON response
                try: 
                    if response.status_code !=204 or response.status_code !=500:
                        json_response = response.json()
                    else:
                        json_response = {}
                except:
                    f.write(image_path)
                    pass

                # Save the response JSON with the same name as the image file
                json_filename = os.path.splitext(image_file)[0] + ".json"
                json_path = os.path.join(image_directory, json_filename)
                
                with open(json_path, 'w') as json_file:
                    json.dump(json_response, json_file, indent=4)

                print(f"Saved JSON response for {image_file} as {json_filename}")
