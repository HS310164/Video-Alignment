import torchvision.transforms as transforms
def transform_images():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    return transform
