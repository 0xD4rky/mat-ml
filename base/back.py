from PIL import Image
import numpy as np

def remove_black_mask_and_join(wall, floor):
    wall_array = np.array(wall)
    floor_array = np.array(floor)
    wall_mask = ~(np.all(wall_array == [255, 255, 255, 0], axis=-1))
    floor_mask = ~(np.all(floor_array == [255, 255, 255, 0], axis=-1))
    wall_filtered = wall_array[wall_mask].reshape((-1, wall_array.shape[1], 4))
    floor_filtered = floor_array[floor_mask].reshape((-1, floor_array.shape[1], 4))
    combined_array = np.vstack((wall_filtered, floor_filtered))
    combined_image = Image.fromarray(combined_array)
    return combined_image

def scale_and_center_image(image, background, scale_factor, vertical_offset=0, horizontal_offset=0):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    center_x = (background.width - new_width) // 2 + horizontal_offset
    center_y = (background.height - new_height) // 2 + vertical_offset
    return resized_image, (center_x, center_y)

def align_shadow(car_mask, shadow_mask, car_y):
    car_mask_array = np.array(car_mask)
    car_height = car_mask_array.shape[0]
    middle_car_y = car_y + car_height // 2
    return middle_car_y

def place_car_and_shadow(background, car, car_mask, shadow_mask):
    car_scaled, (car_x, car_y) = scale_and_center_image(car, background, 1.75, vertical_offset=300) 
    car_mask_scaled, _ = scale_and_center_image(car_mask, background, 1.75)
    car_mask_array = np.array(car_mask_scaled)
    binary_car_mask = np.where(car_mask_array > 128, 255, 0).astype(np.uint8)
    car_masked = Image.composite(car_scaled, Image.new("RGBA", car_scaled.size), Image.fromarray(binary_car_mask, 'L'))

    shadow_scaled, (shadow_x, shadow_y) = scale_and_center_image(shadow_mask, background, 1.75, vertical_offset=675) 
    shadow_alpha = np.array(shadow_scaled.convert('L'))
    shadow_alpha = np.where(shadow_alpha > 128, 128, 0).astype(np.uint8)
    shadow_rgba = np.stack([shadow_alpha]*3 + [shadow_alpha], axis=-1)
    shadow_image = Image.fromarray(shadow_rgba, 'RGBA')

    middle_car_y = align_shadow(car_mask_scaled, shadow_mask, car_y)
    background.paste(car_masked, (car_x, car_y), car_masked)
    background.paste(shadow_image, (shadow_x+60, middle_car_y-80), shadow_image)

    return background

wall_image = Image.open("/Users/darky/Documents/mat-ml/base/assignment/wall.png").convert("RGBA")
floor_image = Image.open("/Users/darky/Documents/mat-ml/base/assignment/floor.png").convert("RGBA")
car_image = Image.open("/Users/darky/Documents/mat-ml/base/assignment/images/1.jpeg").convert("RGBA")
car_mask_image = Image.open("/Users/darky/Documents/mat-ml/base/assignment/car_masks/1.png").convert("L")
shadow_mask_image = Image.open("/Users/darky/Documents/mat-ml/base/assignment/shadow_masks/1.png").convert("RGBA")

background_image = remove_black_mask_and_join(wall_image, floor_image)
final_image = place_car_and_shadow(background_image, car_image, car_mask_image, shadow_mask_image)
final_image.show()
