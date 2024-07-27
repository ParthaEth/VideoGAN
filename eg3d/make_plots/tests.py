import torch
from torchsummary import summary

# Define your PyTorch model
class YourPyTorchModel(torch.nn.Module):
    def __init__(self):
        super(YourPyTorchModel, self).__init__()
        # Your model definition here
        self.a = torch.nn.Linear(3, 2)

    def forward(self, x):
        # Your forward pass logic here
        output = x.sum(dim=1)
        return self.a(output)

# Create an instance of your model
model = YourPyTorchModel()

# Use torchsummary to print model summary and FLOPs
with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
    input_data = torch.randn((3, 3))
    model(input_data)

# Print the profiler results
print(prof)

# Sum up FLOPs from profiler output
total_flops = 0
for event in prof.function_events:
        total_flops += event.flops

print(f"Total FLOPs: {total_flops}")