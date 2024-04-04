from PIL import Image 

def resize_image_to_same_width(image, width):
    """Resize an image to the same width, maintaining the aspect ratio."""
    ratio = width / float(image.width)
    new_height = int(image.height * ratio)
    return image.resize((width, new_height), Image.LANCZOS)

def stack_images_vertically(gt_material, gt_xrd, pred_material, pred_xrd, output_path, width):
    gt_material = resize_image_to_same_width(Image.open(gt_material), width)
    gt_xrd = resize_image_to_same_width(Image.open(gt_xrd), width)
    pred_material = resize_image_to_same_width(Image.open(pred_material), width)
    pred_xrd = resize_image_to_same_width(Image.open(pred_xrd), width)

    assert gt_material.height == pred_material.height
    assert gt_xrd.height == pred_xrd.height
    
    total_height = gt_material.height + gt_xrd.height
    combined_image = Image.new('RGB', (width * 2, total_height))
    
    combined_image.paste(gt_material, (0, 0))
    combined_image.paste(gt_xrd, (0, gt_material.height))
    combined_image.paste(pred_material, (width, 0))
    combined_image.paste(pred_xrd, (width, pred_material.height))
    
    combined_image.save(output_path)

# Example usage
gt_material = '/home/gabeguo/cdvae_xrd/materials_viz/mp_20_tryCompositionInfo/base_truth_material/material0_sample0.png' 
gt_xrd = '/home/gabeguo/cdvae_xrd/materials_viz/mp_20_tryCompositionInfo/base_truth_xrd/material0.png'
pred_material = '/home/gabeguo/cdvae_xrd/materials_viz/mp_20_tryCompositionInfo/opt_material/material0_sample0.png'
pred_xrd = '/home/gabeguo/cdvae_xrd/materials_viz/mp_20_tryCompositionInfo/opt_xrd/material0.png'


output_path = 'dummy_img.png'
desired_width = 500  # Set this to your desired width

stack_images_vertically(gt_material=gt_material,
                        gt_xrd=gt_xrd,
                        pred_material=pred_material,
                        pred_xrd=pred_xrd,
                        output_path=output_path,
                        width=desired_width)
