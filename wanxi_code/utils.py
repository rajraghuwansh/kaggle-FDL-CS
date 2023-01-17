from PIL import Image

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    side = max(img.size)  # Get the longest side of the image
    mask = Image.new('RGB', (side, side), (0, 0, 0))  # Create a square canvas
    mask.paste(img, (0, 0))  # Paste the original image on the left top of the canvas
    mask = mask.resize(size)  # Resize the new image to a uniform size
    return mask

def keep_mask_size_open(path, size=(256, 256)):
    img = Image.open(path)
    side = max(img.size)  # Get the longest side of the image
    mask = Image.new('L', (side, side), 0)  # Create a square canvas
    mask.paste(img, (0, 0))  # Paste the original image on the left top of the canvas
    mask = mask.resize(size)  # Resize the new image to a uniform size
    return mask
