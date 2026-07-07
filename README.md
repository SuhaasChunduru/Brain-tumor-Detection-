# BrainTumorProject

Project structure for brain tumor segmentation/analysis pipeline.

## Structure

- data/BraTS_Dataset: raw BraTS source
- data/cache_npz: processed cache files
- src: preprocessing, training, evaluation scripts
- results: output models, figures, metrics
- notebooks: experiments and analysis

## Quickstart

1. `python -m pip install -r requirements.txt`
2. `python src/preprocess_and_cache.py --input data/BraTS_Dataset --output data/cache_npz`
3. `python src/train_generator.py --data data/cache_npz --models results/models`
4. `python src/evaluate.py --model results/models/best_model.pth --data data/cache_npz`

# Brain-tumor-Deep-Learning-CNN
