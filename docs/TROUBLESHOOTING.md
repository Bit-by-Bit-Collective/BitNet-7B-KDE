# Troubleshooting

## 401/429 from teacher API
- Check `*_API_KEY`; lower concurrency; reduce prompts; raise backoff.

## NaNs in loss
- Ensure KD rows pass mass sanity; drop rows with invalid `other_logprob`.
- Verify `KD_TAU` and next-token alignment; update to latest `losses.py`.

## CUDA OOM
- Lower `TRAIN_BATCH_SIZE`, `MAX_SEQ_LEN`, or model dim/layers.
- Ensure Colab GPU runtime; close extra notebooks.

## Colab Drive not mounting
- Set `AUTO_MOUNT_GDRIVE=1`; confirm `GDRIVE_MOUNT_POINT`; re-run `make ensure_dirs`.
