from torchvision import transforms

class check3c(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

def get_transforms(is_grayscale=False):
    train_transform_list = [
        check3c(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    ]
    test_transform_list = [
        check3c(),
        transforms.Resize((224, 224))
    ]

    if is_grayscale:
        train_transform_list.append(transforms.Grayscale(num_output_channels=3))
        test_transform_list.append(transforms.Grayscale(num_output_channels=3))

    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transform = transforms.Compose(train_transform_list + final_transforms)
    test_transform = transforms.Compose(test_transform_list + final_transforms)
    return train_transform, test_transform
