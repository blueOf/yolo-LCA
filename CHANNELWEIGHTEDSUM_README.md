# ChannelWeightedSum Module

## Overview

`ChannelWeightedSum` is a custom YOLO module that performs channel-wise weighted sum using a learnable C×C weight matrix. This module is designed to maintain the same number of channels (no c2 parameter) while learning optimal channel interactions.

## Design Specification

### Input and Output
- **Input shape**: `(B, C, W, H)`
  - B: Batch size
  - C: Number of channels
  - W: Width
  - H: Height

- **Output shape**: `(B, C, W, H)` (same as input)

### Operation
For each spatial location (w, h) in the feature map:
1. Reshape input from `(B, C, W, H)` to `(B, H*W, 1, C)`
2. Multiply with learnable weight matrix `(C, C)` to compute weighted sum across channels
3. Reshape back to `(B, C, W, H)`

### Mathematical Formula
For each spatial position, the output channels are computed as:
```
output[b, :, w, h] = input[b, :, w, h] @ weight
```
where `@` denotes matrix multiplication.

## Parameters
- **c1** (int): Number of input channels (also the number of output channels)
- **weight** (learnable): C×C weight matrix initialized with small random values

## Usage in YAML Configuration

Since this module doesn't have a c2 parameter (output channels = input channels), it can be used in YAML as:

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # Output: 64 channels
  - [-1, 1, ChannelWeightedSum, []]     # Maintains 64 channels
  - [-1, 1, Conv, [128, 3, 2]]          # Output: 128 channels
  - [-1, 1, ChannelWeightedSum, []]     # Maintains 128 channels
```

### Example: Full Model Configuration

See `test_channelweightedsum_yaml.yaml` for a complete example:

```yaml
nc: 80  # number of classes
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  - [-1, 1, Conv, [256, 3, 2]]      # 2-P3/8
  - [-1, 1, ChannelWeightedSum, []] # 3 - Maintains 256 channels
  - [-1, 1, Conv, [512, 3, 2]]      # 4-P4/16
  - [-1, 1, ChannelWeightedSum, []] # 5 - Maintains 512 channels

head:
  - [-1, 1, Conv, [512, 1, 1]]      # 6
  - [[2, 4, 6], 1, Detect, [nc]]    # 7 - Detect head
```

## Implementation Details

### Module Location
- **File**: `ultralytics/nn/modules/block.py`
- **Class**: `ChannelWeightedSum`

### Integration Points

1. **Module Export** (`ultralytics/nn/modules/__init__.py`):
   - Added to `__all__` list
   - Imported in module namespace

2. **YAML Parsing** (`ultralytics/nn/tasks.py`):
   - Special parsing logic added (lines 2120-2124)
   - Handles modules without c2 parameter
   - Sets `c2 = c1` (output channels = input channels)

### Special Handling in parse_model()

The module requires special parsing because it doesn't follow the standard `(c1, c2, *args)` pattern:

```python
elif m is ChannelWeightedSum:
    # ChannelWeightedSum maintains input channels (no c2 parameter)
    c1 = ch[f]
    c2 = c1  # Output channels = input channels
    args = [c1]  # Only needs c1 parameter
```

## Testing

### Unit Tests
Run `test_channelweightedsum.py` to verify:
- Output shape correctness for various input sizes
- Parameter count (C×C)
- Gradient flow
- Mathematical correctness
- Identity transformation with identity matrix

```bash
python test_channelweightedsum.py
```

### YAML Integration Tests
Run `test_yaml_integration.py` to verify:
- Model loading from YAML
- Forward pass with ChannelWeightedSum layers
- Backward pass and gradient computation
- Integration with other YOLO modules

```bash
python test_yaml_integration.py
```

## Test Results

All tests pass successfully:

### Unit Tests
✓ Single batch, 64 channels: Output shape (1, 64, 32, 32)
✓ Batch of 2, 128 channels: Output shape (2, 128, 16, 16)
✓ Batch of 4, 256 channels: Output shape (4, 256, 8, 8)
✓ RGB-like input: Output shape (1, 3, 224, 224)
✓ Mathematical computation verified
✓ Identity transformation verified

### YAML Integration
✓ Model loaded successfully from YAML
✓ Found 2 ChannelWeightedSum modules
  - model.3: 64 channels, 4,096 parameters (64×64)
  - model.5: 128 channels, 16,384 parameters (128×128)
✓ Forward pass completed successfully
✓ Gradient computed for all modules

## Performance Characteristics

### Parameter Count
- For C channels: C² parameters
- Examples:
  - 64 channels: 4,096 parameters
  - 128 channels: 16,384 parameters
  - 256 channels: 65,536 parameters

### Computational Complexity
- Per spatial location: O(C²) matrix multiplication
- Total: O(B × H × W × C²)

## Use Cases

This module is useful when you want to:
1. Learn channel interactions without changing the number of channels
2. Apply learnable channel-wise transformations
3. Enhance feature representations through weighted channel combinations
4. Maintain architectural constraints where channel count must remain constant

## Comparison with Similar Modules

### vs. 1×1 Convolution
- **1×1 Conv**: Can change number of channels, includes bias and activation
- **ChannelWeightedSum**: Always maintains channel count, pure linear transformation

### vs. Channel Attention (e.g., SENet)
- **Channel Attention**: Uses squeeze-excite with sigmoid activation for per-channel weights
- **ChannelWeightedSum**: Full C×C matrix for richer channel interactions

### vs. Standard Linear Layer
- **Linear Layer**: Typically used on flattened features
- **ChannelWeightedSum**: Operates per spatial location, preserves spatial structure

## Future Enhancements

Potential improvements could include:
1. Optional bias parameter
2. Optional activation function
3. Factorized weight matrix for reduced parameters (e.g., low-rank approximation)
4. Group-wise channel interactions for efficiency

## References

- Problem Statement: "设计一个新的模块... 对于形如B,C,W,H的输入x 进行计算B,H*W,1,C * C,C"
- Implementation: Lines 3114-3161 in `ultralytics/nn/modules/block.py`
- Parsing: Lines 2120-2124 in `ultralytics/nn/tasks.py`
