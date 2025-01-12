# Synthetic

## Requirements

Please follow the installation guide in atari folder.

## Instructions

To test the baseline for input plasticity, run
```
PT_HPU_LAZY_MODE=0 python input_adaptation.py  # Eager Mode
PT_HPU_LAZY_MODE=1 python input_adaptation.py  # Lazy Mode
```

To test the PLASTIC for input plasticity, run
```
PT_HPU_LAZY_MODE=1 python input_adaptation.py --optimizer_type=sam --backbone_norm=ln --policy_reset=True --policy_crelu=True
```
Currently, input_adaptation PLASTIC only works for lazy mode.


To test the baseline for output plasticity, run
```
PT_GPU_LAZY_MODE=1 python output_adaptation.py  # LAZY Mode
```

To test the PLASTIC for output plasticity, run
```
python output_adaptation.py --optimizer_type=sam --backbone_norm=ln --policy_reset=True --policy_crelu=True
```

Currently, output_adaptation baseline & PLASTIC only works for lazy mode.


