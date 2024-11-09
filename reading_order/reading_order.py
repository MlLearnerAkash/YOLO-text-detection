import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_reading_order(bounding_boxes, delta_y=10):
    # Step 1: Sort bounding boxes by `y_min` to group by rows
    bounding_boxes = sorted(bounding_boxes["words"], key=lambda box: box["bounding_box"]['y_min'])
    rows = []
    current_row = [bounding_boxes[0]]
    
    # Step 2: Group boxes into rows based on `delta_y` threshold
    for box in bounding_boxes[1:]:
        if abs(box["bounding_box"]['y_min'] - current_row[-1]["bounding_box"]['y_min']) < delta_y:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)  # Add the last row
    
    # Step 3: Sort each row by `x_min` (left to right)
    ordered_boxes = []
    reading_order = 1
    
    for row in rows:
        row_sorted = sorted(row, key=lambda box: box["bounding_box"]['x_min'])
        for box in row_sorted:
            box['reading_order'] = reading_order
            ordered_boxes.append(box)
            reading_order += 1
    
    return ordered_boxes

def plot_bounding_boxes(bounding_boxes, title="Bounding Boxes", show_order=False):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 200)
    ax.set_ylim(120, 0)  # Invert y-axis to match coordinate layout
    ax.set_aspect('equal')
    ax.axis("off")

    # Draw bounding boxes with optional reading order
    for box_info in bounding_boxes:
        bbox = box_info["bounding_box"]
        text = box_info["text"]
        
        # Draw the rectangle (bounding box)
        rect = patches.Rectangle(
            (bbox["x_min"], bbox["y_min"]),
            bbox["x_max"] - bbox["x_min"],
            bbox["y_max"] - bbox["y_min"],
            linewidth=1, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Display the text and reading order if required
        if show_order:
            order = box_info['reading_order']
            ax.text(
                bbox["x_min"], bbox["y_max"] + 2,
                f"{text} ({order})", color="red", fontsize=8, va='bottom'
            )
        else:
            ax.text(
                bbox["x_min"], bbox["y_max"] + 2,
                text, color="black", fontsize=8, va='bottom'
            )

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Opening the JSON file
    with open('bounding_boxes.json', 'r') as f:
        bounding_boxes = json.load(f)
    
    # Visualize before ordering
    plot_bounding_boxes(bounding_boxes["words"], title="Before Reading Order")

    # Get the ordered boxes and visualize after ordering
    ordered_boxes = get_reading_order(bounding_boxes)
    plot_bounding_boxes(ordered_boxes, title="After Reading Order", show_order=True)
