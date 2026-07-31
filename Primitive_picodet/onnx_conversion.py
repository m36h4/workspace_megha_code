python tools/export_model.py \
    -c configs/picodet/picodet_s_320_mydata.yml \
    -o weights=output/best_model
    
    
paddle2onnx \
    --model_dir output_inference/picodet_s_320_mydata \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --save_file picodet_primitive.onnx \
    --opset_version 13
