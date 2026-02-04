# Owlet Stencil Design with GINNs

## Overview

This integration combines **GINNs** (Geometry-Informed Neural Networks) with **Owlet** (acoustic stencil-based direction-of-arrival estimation) to design optimized 3D-printable acoustic stencils without training data.

## What Was Implemented

### 1. New Problem Definition (`GINN/problems/problem_owlet_stencil.py`)
- Defines cylindrical stencil geometry (13mm radius, 5mm height)
- Base interface (microphone attachment)
- Top surface with holes
- Envelope constraints
- Manufacturing constraints (min hole size, wall thickness)

### 2. Acoustic Simulation (`GINN/acoustic/wave_simulation.py`)
- Simplified acoustic wave propagation model
- Frequency response computation for different angles
- Diffraction modeling through small apertures
- Phase delay from different path lengths
- Angular diversity metrics
- Directional selectivity metrics

### 3. Acoustic Loss Functions (`train/losses_acoustic.py`)
- **Frequency response loss**: Match target frequency characteristics
- **Angular diversity loss**: Maximize distinguishability between angles
- **Directional selectivity loss**: Enhance frontal vs. off-axis separation
- **Hole size constraint**: Ensure manufacturability

### 4. Training Integration
- Updated `train/ginn_trainer.py` to initialize acoustic losses
- Added acoustic loss wrappers (`train/losses_acoustic_wrappers.py`)
- Integrated with existing loss dispatcher system
- Automatic detection of Owlet stencil problem

### 5. Configuration (`configs/GINN/owlet_stencil_wire.yml`)
- Complete training configuration
- Acoustic parameters (frequencies: 500-8000 Hz, angles: 0-180°)
- Loss weights optimized for acoustic objectives
- 2D latent space for design exploration

### 6. Problem Registry
- Updated `util/misc.py` to register `owlet_stencil` problem type

## How It Works

1. **Neural Field**: WIRE network represents stencil as implicit SDF
2. **Geometric Constraints**:
   - Shape must fit within cylinder
   - Attach to base (microphone)
   - Single connected component (no floating parts)
   - Smooth surfaces with sharp holes for diffraction
3. **Acoustic Constraints**:
   - Maximize angular diversity (different angles → different signatures)
   - Enhance directional selectivity
   - Create frequency-dependent responses
4. **Manufacturability**:
   - Minimum hole size (0.5mm)
   - Maximum hole size (2mm)
   - Minimum wall thickness (0.8mm for 3D printing)

## Key Features

- **Data-free**: No training data required, pure constraint optimization
- **Diverse designs**: 2D latent space enables exploration of multiple solutions
- **Physics-informed**: Acoustic simulation guides hole placement and sizing
- **3D printable**: Built-in manufacturability constraints
- **Organized latent space**: Smooth interpolation between designs

## Files Created/Modified

### New Files (10):
1. `GINN/problems/problem_owlet_stencil.py` - Problem definition
2. `GINN/acoustic/__init__.py` - Package init
3. `GINN/acoustic/wave_simulation.py` - Acoustic simulator
4. `train/losses_acoustic.py` - Acoustic loss functions
5. `train/losses_acoustic_wrappers.py` - Loss wrappers for trainer
6. `train/train_utils/loss_calculator_acoustic.py` - Loss calculator
7. `configs/GINN/owlet_stencil_wire.yml` - Configuration
8. `OWLET_INTEGRATION.md` - This file

### Modified Files (3):
1. `util/misc.py` - Added owlet_stencil to problem registry
2. `train/train_utils/loss_keys.py` - Added acoustic loss keys
3. `train/ginn_trainer.py` - Integrated acoustic losses

## Training

The system optimizes for:
- **Objective**: Angular diversity (maximize distinguishability)
- **Constraints**:
  - Geometric (fit cylinder, attach to base, connectedness)
  - Acoustic (directional selectivity, frequency response)
  - Manufacturing (hole sizes, wall thickness)

## Expected Outputs

- Multiple stencil designs with different hole patterns
- Frequency signatures for each design
- STL files for 3D printing
- Acoustic response characteristics

## Next Steps

1. **Run training** to generate stencil designs
2. **Export STL files** from trained models
3. **3D print prototypes**
4. **Real-world testing** with microphones for DoA accuracy
5. **Refinement** based on measured acoustic performance

## Technical Details

- **Model**: WIRE (wavelet-based implicit neural representation)
- **Latent dimension**: 2D for easy exploration
- **Training epochs**: 5000 (adjustable)
- **Optimizer**: Adam with learning rate 0.001
- **Batch size**: 1 shape at a time
- **Surface points**: 16,384 for accurate geometry

## Advantages Over Manual Design

1. **Automated optimization** - No manual trial-and-error
2. **Multiple solutions** - Explore design space systematically
3. **Guaranteed manufacturability** - Built-in constraints
4. **Acoustic properties** - Direct optimization of frequency response
5. **Organized exploration** - Latent space structure for interpolation
