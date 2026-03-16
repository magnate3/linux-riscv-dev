import torch
from torchvision.models import resnet50
import torch_pruning as tp

model = resnet50(pretrained=True).eval()

# 1. Build dependency graph for a resnet18. This requires a dummy input for forwarding
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. To prune the output channels of model.conv1, we need to find the corresponding group with a pruning function and pruning indices.
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

# 3. Do the pruning
if DG.check_pruning_group(group): # avoid over-pruning, i.e., channels=0.
    group.prune()
    print(group.details()) # or print(group)
    
# 4. Save & Load
model.zero_grad() # clear gradients to avoid a large file size
torch.save(model, 'resnet50-prune.pth') # !! no .state_dict here since the structure has been changed after pruning
model = torch.load('resnet50-prune.pth') # load the pruned model. you may need torch.load('model.pth', weights_only=False) for PyTorch 2.6.0+.
