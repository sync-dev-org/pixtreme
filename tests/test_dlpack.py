import torch

import pixtreme as px


def test_dlpack():
    image = px.imread("example/example.png")

    # to_tensor
    tensor = px.to_tensor(image)
    print(f"type(tensor): {type(tensor)}")
    assert isinstance(tensor, torch.Tensor)

    dlpack = image.__dlpack__()

    print(f"type(dlpack): {type(dlpack)}")  # PyCapsule
