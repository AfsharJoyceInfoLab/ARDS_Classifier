# ARDS_Classifier
 
### Run CUI model

```
export CUI_DATA_DIR=/path/to/cui_dataset/
export OUTPUT_DIR=/path/to/outputs/

python3 run_classifier.py \
    --cui_data_dir=CUI_DATA_DIR \
    --output_dir=OUTPUT_DIR
```

### Run text model

```
export TEXT_DATA_DIR=/path/to/text_dataset/
export OUTPUT_DIR=/path/to/outputs/

python3 run_classifier.py \
    --text_data_dir=TEXT_DATA_DIR \
    --output_dir=OUTPUT_DIR
```