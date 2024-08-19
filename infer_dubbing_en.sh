python inference_en.py \
    --lang 'english'\
    --src_speech './sample/va_speech.wav' \
    --src_bgm './sample/va_bgm.wav' \
    --trg_text './sample/va_txt' \
    --ckpt_model './ckpt/baseline.pth' \
    --config './ckpt/config.json' \
    --output_dir './dubbed' \