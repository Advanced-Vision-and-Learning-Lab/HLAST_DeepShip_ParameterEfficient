# Histogram-based Parameter-efficient Tuning for Passive Sonar Classification
<p align="center">
  <img src="Figures/Workflow.png" alt="Workflow Diagram">
</p>


Amirmohammad Mohammadi, Davelle Carreiro, Alexandra Van Dine and Joshua Peeples

If this code is used, please cite it. (2025, March): Initial Release (Version v1.0). 

[arXiv](https://arxiv.org/abs/2504.15214)

### Zenodo archive

[`Zenodo`](https://doi.org/10.5281/zenodo.15263760). https://zenodo.org/records/15263760  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15263760.svg)](https://doi.org/10.5281/zenodo.15263760)

## Installation Prerequisites

The [`requirements.txt`](requirements.txt) file includes all the necessary packages, and the packages will be installed using:

   ```pip install -r requirements.txt```

Or, for a more convenient option, a pre-configured environment is available for download through the provided link. This environment comes with all the essential packages already installed.

[`Download environment`](https://drive.google.com/file/d/1dUvjjK3ImNZJP6jZuAlMhulKfijVkM8K/view?usp=sharing)

## Demo

To get started, please follow the instructions in the [Datasets](Datasets) folder to download the DeepShip dataset.
Next, run [`demo.py`](demo.py) in Python IDE (e.g., Spyder) or command line to train, validate, and test models. 


## Inventory

```
https://github.com/Peeples-Lab/HLAST_DeepShip_ParameterEfficient 

└── root directory
    ├── demo_light.py                     // Main demo file.
    ├── Demo_Parameters.py                // Parameter file for the demo.
    ├── plot_curves.py                    // Run this after the demo to view learning curves. 
    ├── feature_similarity_analysis.py    // Run this after the demo to view feature similarites, PLEASE set the parameters accordingly. 
    └── Datasets                
        ├── Get_Preprocessed_Data.py       // Generate segments for the DeepShip dataset.
        └── SSDataModule.py                // Data Module for the DeepShip dataset.
        ├── ShipsEar_Data_Preprocessing.py // Generate segments for the ShipsEar dataset.
        └── ShipsEar_dataloader.py         // Data Module for the ShipsEar dataset.
        ├── Create_Combined_VTUAD.py 	   // Merge the three distinct scenarios into one for the VTUAD dataset.
        └── VTUAD_DataModule.py            // Data Module for the VTUAD dataset.
    └── Utils                     
        ├── LitModel.py                    // Lightning Module for the the model.
        ├── Network_functions.py           // Contains functions to initialize the model.
        ├── LogMelFilterBank.py            // Log Mel Filter Bank Feature.
        └── Feature_Extraction_Layer.py    // Extract and transform features from the audio files.
    └── src
    	└── models              
		├── ast_base.py            // AST Original Model
		├── ast_linear_probe.py    // AST Linear Probing
		├── ast_adapter.py         // AST with Adapter Layers
		├── RBFHistogramPooling.py // Create the Histogram Layer
		└── ast_histogram.py       // AST with Histogram Layers (HPT)

```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2025 A. Mohammadi, D. Carreiro, A. Dine and J. Peeples. All rights reserved.

## Citation

If you use this work, please cite:

### arXiv preprint

```bibtex
@article{amir2025histogram,
  title={Histogram-based Parameter-efficient Tuning for Passive Sonar Classification},
  author={Mohammadi, Amirmohammad and Carreiro, Davelle and Van Dine, Alexandra and Peeples, Joshua},
  journal={arXiv preprint arXiv:2504.15214},
  year={2025}
}
