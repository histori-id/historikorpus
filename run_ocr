from huggingface_hub import hf_hub_download
import torch
from doclayout_yolo import YOLOv10
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from google.colab import drive
import os
import pytesseract

# Download the pre-trained model from Hugging Face
model_path = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
input_folder = "/content/drive/MyDrive/tugas-akhir/koran-lain/left-images"  
output_folder = "/content/drive/MyDrive/tugas-akhir/koran-lain/ocr-results"  

# Load the model
model = YOLOv10(model_path)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
total_files = len(image_files)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for idx, image_file in enumerate(image_files):

    output_image_path = os.path.join(output_folder, f"appearance_order_{image_file}")

    # Load image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict
    det_res = model.predict(image_path, imgsz=2048, conf=0.2, device="cuda" if torch.cuda.is_available() else "cpu")

    # Extract bounding boxes and labels
    bboxes = det_res[0].boxes.xyxy.cpu().numpy()
    labels = det_res[0].boxes.cls.cpu().numpy()
    categories = [model.names[int(label)] for label in labels]

    # Separate title and plain text boxes
    titles, plain_texts = [], []

    for bbox, category in zip(bboxes, categories):
        if category == "title":
            titles.append(bbox)
        elif category == "plain text":
            plain_texts.append(bbox)

    # Sort title boxes top to bottom, then left to right
    titles = sorted(titles, key=lambda b: (b[1], b[0]))

    # Assuming titles and plain_texts are predefined lists of box coordinates
    boxes = titles + plain_texts
    boxes = [list(map(int, b)) for b in boxes]

    # Compute column_width
    text_widths = [box[2] - box[0] for box in titles]
    column_width = np.median(text_widths) if text_widths else 0

    def is_aligned(upper, lower, obstacles):
        ux = (upper[0] + upper[2]) / 2
        lx = (lower[0] + lower[2]) / 2

        top = upper[3]
        bottom = lower[1]
        if not (lower[0] <= ux <= lower[2] or upper[0] <= lx <= upper[2]):
            return False
        for ob in obstacles:
            ob_left, ob_top, ob_right, ob_bottom = ob
            if ob_left <= ux <= ob_right and ob_top < bottom and ob_bottom > top:
                return False
        return True

    # Grouping titles with enhancement for titles above wide titles
    title_sets, used_indices = [], set()

    for i, upper in enumerate(titles):
        if i in used_indices:
            continue

        current_set = [upper]
        used_indices.add(i)

        for j in range(i + 1, len(titles)):
            if j in used_indices:
                continue

            lower = titles[j]
            if is_aligned(upper, lower, plain_texts):
                current_set.append(lower)
                used_indices.add(j)
                upper = lower

                is_wide_title = (lower[2] - lower[0]) >= 1.3 * column_width
                if is_wide_title:
                    for k, prev_title in enumerate(titles[:i]):
                        if k not in used_indices and is_aligned(prev_title, lower, plain_texts):
                            current_set.append(prev_title)
                            used_indices.add(k)

        title_sets.append(current_set)

    # Identify wide titles and represented titles
    wide_titles, represented_titles = [], []

    for tset in title_sets:
        tset = sorted(tset, key=lambda b: b[1])
        widths = [box[2] - box[0] for box in tset]
        is_wide_set = any(width >= 1.3 * column_width for width in widths)
        if is_wide_set:
            wide_titles.append(tset[-1])
        represented_titles.extend(tset[:-1])


    def treat_as_wide_or_short(represented_title):
        set_has_wide = any(
            any(np.array_equal(represented_title, box) for box in tset)
            for tset in title_sets
            if any(np.array_equal(w, wt) for w in tset for wt in wide_titles)
        )
        return "wide" if set_has_wide else "short"

    # Horizontal line detection
    def detect_horizontal_lines(image, min_line_length=30, max_line_gap=5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 10)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        dilated = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

    def filter_lines_overlapping_boxes(lines, boxes):
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y_bar = (y1 + y2) / 2
            overlap = False
            #for box in plain_texts:
            for box in boxes:
                left, top, right, bottom = box
                if (left <= x1 <= right or left <= x2 <= right) and (top <= y_bar <= bottom):
                    overlap = True
                    break
            if not overlap:
                filtered_lines.append(line)
        return filtered_lines

    def is_short(box):
        width = box[2] - box[0]
        return width < 1.3 * column_width and not any(np.array_equal(box, rep) for rep in represented_titles)

    def is_wide(box):
        return (
            any(np.array_equal(box, wt) for wt in wide_titles) or
            (any(np.array_equal(box, rep) for rep in represented_titles) and treat_as_wide_or_short(box) == "wide")
        )

    def box_in_list(target_box, box_list):
        return any(np.array_equal(target_box, box) for box in box_list)

    def direction_between(top, bottom):
        if box_in_list(top, plain_texts):
            if box_in_list(bottom, plain_texts):
                return "LEFT"
            elif is_wide(bottom):
                return "END"
            elif is_short(bottom):
                return "UP"
        elif is_wide(top):
            if box_in_list(bottom, plain_texts):
                return "UP"
            elif is_short(bottom):
                return "END"
        elif is_short(top):
            if box_in_list(bottom, plain_texts):
                return "UP"
            elif is_wide(bottom):
                return "NONE"
        return None

    def find_directly_above_below(y_bar, x1, x2, boxes, max_dist=300):
        directly_above = None
        directly_below = None
        min_above_dist = float('inf')
        min_below_dist = float('inf')

        for box in boxes:
            left, top, right, bottom = box
            if not (left <= x1 <= right or left <= x2 <= right):
                continue

            if bottom <= y_bar:
                dist = y_bar - bottom
                if dist < min_above_dist and dist <= max_dist:
                    directly_above = box
                    min_above_dist = dist
            elif top >= y_bar:
                dist = top - y_bar
                if dist < min_below_dist and dist <= max_dist:
                    directly_below = box
                    min_below_dist = dist

        return directly_above, directly_below

    # Adding merge logic for left linked blocks
    def get_box_centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def euclidean_distance(centroid1, centroid2):
        return np.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)

    def find_and_merge_with_left_link(group_indices, noise_filtered_boxes, trigger_box_idx, groups_of_indices):
        trigger_box = noise_filtered_boxes[trigger_box_idx]
        trigger_centroid = get_box_centroid(trigger_box)

        best_left_block_idx = -1
        min_left_distance = float('inf')
        merge_target_group_idx = -1

        for other_idx, other_box in enumerate(noise_filtered_boxes):
            if other_idx == trigger_box_idx or other_idx in group_indices:
                continue

            containing_group_idx = -1
            for g_idx, group in enumerate(groups_of_indices):
                if other_idx in group:
                    containing_group_idx = g_idx
                    break

            if containing_group_idx == -1:
                continue

            other_centroid = get_box_centroid(other_box)
            if other_centroid[0] < trigger_centroid[0]:
                distance = euclidean_distance(trigger_centroid, other_centroid)
                if distance < min_left_distance:
                    min_left_distance = distance
                    best_left_block_idx = other_idx
                    merge_target_group_idx = containing_group_idx

        if best_left_block_idx != -1 and merge_target_group_idx != -1:
            print(f"Merging current group with group {merge_target_group_idx} due to left link with block {best_left_block_idx}.")
            for idx in group_indices:
                if idx not in groups_of_indices[merge_target_group_idx]:
                    groups_of_indices[merge_target_group_idx].append(idx)
            return True

        return False

    # Link boxes separated by horizontal bars
    lines = detect_horizontal_lines(image)
    all_boxes = titles + plain_texts
    filtered_lines = filter_lines_overlapping_boxes(lines, all_boxes)
    boxes = plain_texts + titles
    links = set()

    output_image = image_rgb.copy()

    if filtered_lines is None:
        filtered_lines = []

    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        y_bar = (y1 + y2) / 2

        top_box, bottom_box = find_directly_above_below(y_bar, x1, x2, boxes)
        if top_box is not None and bottom_box is not None:
            dir_code = direction_between(top_box, bottom_box)
            if dir_code and ((tuple(top_box), tuple(bottom_box)) not in links):
                links.add((tuple(top_box), tuple(bottom_box), dir_code))
                #cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    def compute_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0  # No intersection

        return (x_right - x_left) * (y_bottom - y_top)

    def get_noise_filtered_boxes(boxes):
        retain_indices = set(range(len(boxes)))

        for i, box in enumerate(boxes):
            if i not in retain_indices:
                continue
            area_i = compute_area(box)

            for j, other_box in enumerate(boxes):
                if i == j or j not in retain_indices:
                    continue

                overlap_area = intersection_area(box, other_box)
                overlap_ratio_i = overlap_area / area_i if area_i > 0 else 0
                overlap_ratio_j = overlap_area / compute_area(other_box) if compute_area(other_box) > 0 else 0

                if overlap_ratio_i >= 0.75 or overlap_ratio_j >= 0.75:
                    if area_i >= compute_area(other_box):
                        retain_indices.discard(j)
                    else:
                        retain_indices.discard(i)
                        break

        return [boxes[i] for i in retain_indices]

    # Get noise-filtered boxes
    noise_filtered_boxes = get_noise_filtered_boxes(boxes)

    def box_centroid_x(box):
        return (box[0] + box[2]) / 2

    def calculate_centroid(box):
        x1, y1, x2, y2 = box
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        return (centroid_x, centroid_y)

    def is_above(b1, b2):
        b1_x1, b1_y1, b1_x2, _ = b1
        b2_x1, b2_y1, b2_x2, _ = b2
        x_c_b1 = box_centroid_x(b1)
        x_c_b2 = box_centroid_x(b2)

        horizontally_aligned = (b1_x1 <= x_c_b2 <= b1_x2) or (b2_x1 <= x_c_b1 <= b2_x2)
        vertically_aligned = b2_y1 <= b1_y1

        return horizontally_aligned and vertically_aligned

    def show_blocks_with_left_links(links, noise_filtered_boxes, centroid_to_group_map):
        link_index = 1  # Start numbering links from 1
        printing_space = 0

        for top, bottom, direction in links:
            if direction == "LEFT":
                bottom_centroid = calculate_centroid(bottom)
                left_block_idx = None
                min_horizontal_distance = float('inf')
                x_min, y_min, x_max, y_max = map(int, bottom)

                for idx, box in enumerate(noise_filtered_boxes):
                    box_centroid = calculate_centroid(box)
                    horizontal_distance = bottom_centroid[0] - box_centroid[0]

                    if box_centroid[0] < bottom_centroid[0]:
                        candidate_y_min, candidate_y_max = box[1], box[3]

                        if ((y_min <= box_centroid[1] <= y_max) or
                            (candidate_y_min <= bottom_centroid[1] <= candidate_y_max)):
                            if horizontal_distance < min_horizontal_distance:
                                min_horizontal_distance = horizontal_distance
                                left_block_idx = idx

                if left_block_idx is not None:
                    left_block = noise_filtered_boxes[left_block_idx]
                    left_centroid = calculate_centroid(left_block)
                    left_block_group_number = centroid_to_group_map.get(left_centroid, None)

                    # Visualize bottom block
                    x_min, y_min, x_max, y_max = map(int, bottom)
                    #cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red for bottom block
                    # Put link index on the bottom block
                    #cv2.putText(output_image, str(link_index), (x_min + printing_space , y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)

                    # Visualize left block
                    x1, y1, x2, y2 = map(int, left_block)
                    #cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange for left block
                    # Put link index on the left block
                    #cv2.putText(output_image, str(link_index), (x1 + printing_space, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
                    #cv2.putText(output_image, str(left_block_group_number), (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    # Draw the line connecting centroids
                    #cv2.line(output_image,
                            #(int(bottom_centroid[0]), int(bottom_centroid[1])),
                            #(int(left_centroid[0]), int(left_centroid[1])),
                            #(0, 255, 0), 2)  # Green line

                    link_index += 1  # Increment link number for the next pair
                    printing_space += 20

    # Initialize grouping
    assigned = set()
    groups, groups_of_indices = [], []
    centroid_to_group_map = {}  # Map from centroid to group index
    grouping_details = []  # To store the group information as requested

    image_height, image_width = image.shape[:2]

    start_from_top_box = False
    next_start_idx = None

    def is_title_box(index):
        box = noise_filtered_boxes[index]
        return any(np.array_equal(box, title) for title in titles)

    def is_wide_paragraph_box(index):
        box = noise_filtered_boxes[index]
        box_width = box[2]-box[0]

        if box_width < 1.3 * column_width:
            return False
        else:
            return any(np.array_equal(box, plain_text) for plain_text in plain_texts)

    def is_plain_text_box(index):
        box = noise_filtered_boxes[index]
        return any(np.array_equal(box, plain_text) for plain_text in plain_texts)

    def calculate_top_left_distance(centroid):
        return np.sqrt(centroid[0]**2 + centroid[1]**2)

    # Grouping logic
    while len(assigned) < len(noise_filtered_boxes):
        if start_from_top_box and next_start_idx is not None:
            bottom_left_idx = next_start_idx
            start_from_top_box = False
        else:
            min_distance = float('inf')
            bottom_left_idx = None

            for idx, box in enumerate(noise_filtered_boxes):
                if idx in assigned:
                    continue

                x1, y1, x2, y2 = box
                distance = np.sqrt((x1 - 0) ** 2 + (y2 - image_height) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    bottom_left_idx = idx

        if bottom_left_idx is None:
            break

        # Start a new group
        current_group_indices = []
        current = bottom_left_idx
        should_continue_grouping = True

        while current is not None and should_continue_grouping:
            current_group_indices.append(current)
            assigned.add(current)

            # Check each link for the current block
            for (top, bottom, direction) in links:
                if direction == "LEFT" and np.array_equal(noise_filtered_boxes[current], bottom):

                    # Find the block directly above the bottom block, if such exists
                    directly_above = None
                    min_distance = float('inf')
                    bottom_centroid = calculate_centroid(bottom)

                    for j, other_box in enumerate(noise_filtered_boxes):
                        if j == current:  # Avoid self
                            continue

                        other_centroid = calculate_centroid(other_box)

                        # Check if other_box is directly above
                        if ((bottom_centroid[0] >= other_box[0] and bottom_centroid[0] <= other_box[2]) or
                            (other_centroid[0] >= bottom[0] and other_centroid[0] <= bottom[2])):
                            distance = bottom_centroid[1] - other_centroid[1]

                            if distance > 0 and distance < min_distance:
                                min_distance = distance
                                directly_above = j

                    # If there's a directly above block, check if it's a title
                    if directly_above is not None and is_title_box(directly_above):
                        break  # Exit the for loop if the directly above block is a title

                    should_continue_grouping = False

                    x_min, y_min, x_max, y_max = map(int, bottom)

                    # Identify and merge with the left block's group
                    bottom_centroid = calculate_centroid(bottom)
                    left_block_idx = None
                    min_horizontal_distance = float('inf')

                    for idx, box in enumerate(noise_filtered_boxes):
                        box_centroid = calculate_centroid(box)
                        if box_centroid[0] < bottom_centroid[0]:
                            candidate_y_min, candidate_y_max = box[1], box[3]
                            if ((y_min <= box_centroid[1] <= y_max) or
                                (candidate_y_min <= bottom_centroid[1] <= candidate_y_max)):
                                horizontal_distance = bottom_centroid[0] - box_centroid[0]
                                if horizontal_distance < min_horizontal_distance:
                                    min_horizontal_distance = horizontal_distance
                                    left_block_idx = idx

                    # Look up the group number for the left block
                    if left_block_idx is not None:
                        left_block_centroid = calculate_centroid(noise_filtered_boxes[left_block_idx])
                        left_block_group_number = centroid_to_group_map.get(left_block_centroid, None)

                        # Reassign current blocks to this group
                        if left_block_group_number is not None:
                            groups_of_indices[left_block_group_number].extend(current_group_indices)
                            groups_of_indices[left_block_group_number] = list(set(groups_of_indices[left_block_group_number]))
                            for idx in current_group_indices:
                                centroid = calculate_centroid(noise_filtered_boxes[idx])
                                centroid_to_group_map[centroid] = left_block_group_number
                        else:
                            groups.append([noise_filtered_boxes[i] for i in current_group_indices])
                            groups_of_indices.append(current_group_indices)

                            group_index = len(groups_of_indices) - 1
                            for idx in current_group_indices:
                                centroid = calculate_centroid(noise_filtered_boxes[idx])
                                centroid_to_group_map[centroid] = group_index

                            # Calculate the distance of the centroid of the last processed block to the top-left corner
                            last_processed_index = current_group_indices[-1]
                            topmost_centroid = calculate_centroid(noise_filtered_boxes[last_processed_index])
                            distance_to_top_left = calculate_top_left_distance(topmost_centroid)

                            # Store the triple (distance to top-left, group index, block index)
                            grouping_details.append((distance_to_top_left, group_index, last_processed_index))
                    else :
                        groups.append([noise_filtered_boxes[i] for i in current_group_indices])
                        groups_of_indices.append(current_group_indices)

                        group_index = len(groups_of_indices) - 1
                        for idx in current_group_indices:
                            centroid = calculate_centroid(noise_filtered_boxes[idx])
                            centroid_to_group_map[centroid] = group_index

                        # Calculate the distance of the centroid of the last processed block to the top-left corner
                        last_processed_index = current_group_indices[-1]
                        topmost_centroid = calculate_centroid(noise_filtered_boxes[last_processed_index])
                        distance_to_top_left = calculate_top_left_distance(topmost_centroid)

                        # Store the triple (distance to top-left, group index, block index)
                        grouping_details.append((distance_to_top_left, group_index, last_processed_index))

                    # Prepare to continue from the top box of the link
                    indices = np.where([np.array_equal(top, b) for b in noise_filtered_boxes])[0]
                    if len(indices) > 0:
                        next_start_idx = indices[0]
                        start_from_top_box = True
                    else:
                        next_start_idx = None
                    break

                elif direction == "END" and np.array_equal(noise_filtered_boxes[current], bottom):
                    # Endpoint, finalize current group and proceed
                    should_continue_grouping = False
                    break

                elif direction == "UP" and np.array_equal(noise_filtered_boxes[current], bottom):

                    directly_above = None
                    min_distance = float('inf')
                    bottom_centroid = calculate_centroid(bottom)

                    for j, other_box in enumerate(noise_filtered_boxes):
                        if j == current:  # Avoid self
                            continue

                        other_centroid = calculate_centroid(other_box)

                        # Check if other_box is directly above
                        if ((bottom_centroid[0] >= other_box[0] and bottom_centroid[0] <= other_box[2]) or
                            (other_centroid[0] >= bottom[0] and other_centroid[0] <= bottom[2])):
                            distance = bottom_centroid[1] - other_centroid[1]

                            if distance > 0 and distance < min_distance:
                                min_distance = distance
                                directly_above = j

                    # If there's a directly above block, check if it's a title
                    if directly_above is not None and is_wide_paragraph_box(directly_above):
                        should_continue_grouping = False
                        break  # Exit the for loop if the directly above block is a title

            if should_continue_grouping:
                next_above = None
                min_vertical_distance = float('inf')

                # Define the maximum allowable vertical distance (for example, this can be arbitrary, adjust as necessary).
                max_vertical_distance_limit = 150  

                for j in range(len(noise_filtered_boxes)):
                    if j not in assigned and is_above(noise_filtered_boxes[current], noise_filtered_boxes[j]):
                        vertical_distance = noise_filtered_boxes[current][1] - noise_filtered_boxes[j][3]
                        if vertical_distance < min_vertical_distance:
                            min_vertical_distance = vertical_distance
                            next_above = j

                # Check if the minimum vertical distance exceeds the maximum limit
                if min_vertical_distance >= max_vertical_distance_limit:
                    should_continue_grouping = False
                else:
                    current = next_above

        if not start_from_top_box and len(current_group_indices) > 0:  # Finalize group if "END" or normal path
            groups.append([noise_filtered_boxes[i] for i in current_group_indices])
            groups_of_indices.append(current_group_indices)

            group_index = len(groups_of_indices) - 1
            for idx in current_group_indices:
                centroid = calculate_centroid(noise_filtered_boxes[idx])
                centroid_to_group_map[centroid] = group_index

            # Calculate the distance of the centroid of the last processed block to the top-left corner
            last_processed_index = current_group_indices[-1]
            topmost_centroid = calculate_centroid(noise_filtered_boxes[last_processed_index])
            distance_to_top_left = calculate_top_left_distance(topmost_centroid)

            # Store the triple (distance to top-left, group index, block index)
            grouping_details.append((distance_to_top_left, group_index, last_processed_index))

    # Sort triples based on the distance
    grouping_details.sort(key=lambda x: x[0])

    cmap = plt.colormaps.get_cmap("tab20")
    colors = [tuple(int(x * 255) for x in cmap(i / len(groups))[:3]) for i in range(len(groups))]

    # After the main processing loop and before final visualization:
    # show_blocks_with_left_links(links, noise_filtered_boxes, centroid_to_group_map)

    for box in wide_titles:
        x_min, y_min, x_max, y_max = map(int, box)
        label_x = x_max - 10
        label_y = y_min + 50
        #cv2.putText(output_image, "WIDE", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        #cv2.putText(output_image, "WIDE", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    for top, bottom, direction in links:
        cx1, cy1 = int((top[0] + top[2]) / 2), int((top[1] + top[3]) / 2)
        cx2, cy2 = int((bottom[0] + bottom[2]) / 2), int((bottom[1] + bottom[3]) / 2)
        #cv2.line(output_image, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        label_pos = (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))
        #cv2.putText(output_image, direction, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        #cv2.putText(output_image, direction, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    #print("Centroid to Group Map:")
    #for centroid, group_idx in centroid_to_group_map.items():
    #    print(f"Centroid {centroid}: Group {group_idx}")


    def calculate_centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def cluster_by_horizontal_position(noise_filtered_boxes, groups_of_indices, output_image):

        extracted_text = []

        output_folder_image = os.path.join(output_folder, os.path.splitext(image_file)[0])
        os.makedirs(output_folder_image)

        #for group_id, indices in enumerate(groups_of_indices):
        for order_index, (_, group_id, _) in enumerate(grouping_details, start=1):

            indices = groups_of_indices[group_id]

            x_coords = np.array([noise_filtered_boxes[idx][0] for idx in indices]).reshape(-1, 1)

            if x_coords.size == 0:
                continue

            dbscan = DBSCAN(eps=200, min_samples=1)
            cluster_labels = dbscan.fit_predict(x_coords)

            cmap = plt.get_cmap("tab10")
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_clusters)]

            # Dictionary to store blocks by cluster index
            clusters = {i: [] for i in range(num_clusters)}

            # Group blocks by cluster labels
            for idx, label in zip(indices, cluster_labels):
                if label != -1:
                    clusters[label].append(noise_filtered_boxes[idx])

            # Sort clusters by the x-coordinate of their leftmost block
            sorted_clusters = sorted(clusters.items(), key=lambda c: min([box[0] for box in c[1]]))

            group_text = []

            output_folder_partition = os.path.join(output_folder_image, str(order_index))
            os.makedirs(output_folder_partition)

            block_counter = 1

            for _, blocks in sorted_clusters:
                # Sort blocks within each cluster by their top position (y-coordinate)
                sorted_blocks = sorted(blocks, key=lambda b: b[1])

                """
                for block_index, bbox in enumerate(sorted_blocks, start=1):

                    block_text = []

                    x_min, y_min, x_max, y_max = map(int, bbox)
                    # Extract the region of interest
                    roi = image_rgb[y_min:y_max, x_min:x_max]
                    roi_pil = Image.fromarray(roi)
                    # Perform OCR
                    text = pytesseract.image_to_string(roi_pil, lang="ind+eng+nld").strip()
                    block_text.append(text)

                    block_output_file = os.path.join(output_folder_partition, f"block_{block_index}_{image_file}.txt")
                    with open(block_output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(block_text))

                    group_text.append(text)
                    extracted_text.append(text)
                """

                # Initialize the counter before the loop

                for bbox in sorted_blocks:
                    block_text = []

                    x_min, y_min, x_max, y_max = map(int, bbox)
                    # Extract the region of interest
                    roi = image_rgb[y_min:y_max, x_min:x_max]
                    roi_pil = Image.fromarray(roi)
                    # Perform OCR
                    text = pytesseract.image_to_string(roi_pil, lang="ind+eng+nld").strip()
                    block_text.append(text)

                    block_output_file = os.path.join(output_folder_partition, f"block_{block_counter}_{image_file}.txt")
                    with open(block_output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(block_text))

                    group_text.append(text)
                    extracted_text.append(text)

                    # Increment the counter
                    block_counter += 1


            partition_output_file = os.path.join(output_folder_partition, f".partition_{order_index}_{image_file}.txt")
            with open(partition_output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(group_text))


        # Save extracted text to file
        output_file = os.path.join(output_folder_image, f"article_set_{image_file}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(extracted_text))

    def visualize_grouping(groups_of_indices, noise_filtered_boxes, output_image):
        # Assign colors to each group for visualization
        cmap = plt.get_cmap("tab20")
        num_groups = len(groups_of_indices)
        colors = [tuple(int(x * 255) for x in cmap(i % 20)[:3]) for i in range(num_groups)]

        for group_id, indices in enumerate(groups_of_indices):
            color = colors[group_id]
            for idx in indices:
                x1, y1, x2, y2 = noise_filtered_boxes[idx]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                #cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                cx, cy = map(int, calculate_centroid(noise_filtered_boxes[idx]))
                # Display group information
                # cv2.putText(output_image, f"Set {group_id}", (x2 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        for order_index, (distance, group_id, block_index) in enumerate(grouping_details):
            color = colors[group_id]
            x1, y1, x2, y2 = noise_filtered_boxes[block_index]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            # Place a very large number showing the order in the sorted list
            order_number = str(order_index + 1)   # Repeat to make large enough
            # cv2.putText(output_image, order_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 3)

            #cv2.putText(output_image, f"Set {group_id}", (x2 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

        cluster_by_horizontal_position(noise_filtered_boxes, groups_of_indices, output_image)

        # Save the result with correct color
        # cv2.imwrite(output_image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    removed_groups = set()
    grouping_details.reverse()

    for i, (distance, group_id, block_index) in enumerate(grouping_details):
        if group_id in removed_groups:
            continue

        # Topmost block's boundaries
        topmost_block = noise_filtered_boxes[block_index]
        topmost_centroid = calculate_centroid(topmost_block)

        x1, x2 = topmost_block[0], topmost_block[2]

        # Highlight the topmost block with a big circle
        cx, cy = map(int, topmost_centroid)

        # Check for existing links using np.array_equal for proper comparison
        has_link = any(np.array_equal(topmost_block, bottom) for top, bottom, _ in links)

        above_block_info = None

        if not has_link:
            # No link found, search for block directly above
            directly_above = None
            min_distance = float('inf')

            for j, other_box in enumerate(noise_filtered_boxes):
                if j == block_index or centroid_to_group_map.get(calculate_centroid(other_box)) in removed_groups:
                    continue

                other_centroid = calculate_centroid(other_box)

                # Check horizontal alignment and above condition
                if ((topmost_centroid[0] >= other_box[0] and topmost_centroid[0] <= other_box[2]) or
                    (other_centroid[0] >= topmost_block[0] and other_centroid[0] <= topmost_block[2])):
                    distance = topmost_centroid[1] - other_centroid[1]

                    if distance > 0 and distance < min_distance:  # Ensuring it's above the current block
                        min_distance = distance
                        directly_above = j

            if directly_above is not None:
                above_block = noise_filtered_boxes[directly_above]
                above_group_id = centroid_to_group_map[calculate_centroid(above_block)]
                above_block_info = (above_block[0], above_block[2], above_group_id)

                # Highlight the above block's centroid
                ax, ay = map(int, calculate_centroid(above_block))
                # cv2.circle(output_image, (ax, ay), 10, (0, 255, 0), 2)  # Green circle for above block centroid

                # Reallocate group
                groups_of_indices[above_group_id].extend(groups_of_indices[group_id])
                groups_of_indices[above_group_id] = list(set(groups_of_indices[above_group_id]))

                # Update centroid mapping
                for idx in groups_of_indices[group_id]:
                    centroid = calculate_centroid(noise_filtered_boxes[idx])
                    centroid_to_group_map[centroid] = above_group_id

                # Remove old group
                removed_groups.add(group_id)
                groups_of_indices[group_id] = []  # Clear group indices
                grouping_details[i] = None  # Mark for removal

    # Remove cleared groups from details
    grouping_details = [detail for detail in grouping_details if detail is not None]

    grouping_details.sort(key=lambda x: x[0])

    # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Visualization of the result
    visualize_grouping(groups_of_indices, noise_filtered_boxes, output_image)

plt.close('all')
