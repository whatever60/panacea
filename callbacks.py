import pytorch_lightning as pl


class LogPredictionsCallback(pl.Callback):
    def custom_on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        """log 20 sample image predictions from first batch"""
        if batch_idx != 0:
            return
        n = 8
        _, *labels, _ = batch
        xs = outputs["images"]
        generated = outputs["generated_images"]

        if len(labels) == 1:
            labels = labels[0]
            captions_real = [f"Class {l} - real" for l in labels]
            captions_fake = [f"Class {l} - generated" for l in labels]
        else:
            ls, bs = labels[1:3]
            captions_real = [f"Number {l} of batch {b} - real" for l, b in zip(ls, bs)]
            captions_fake = [
                f"Number {l} of batch {b} - generated" for l, b in zip(ls, bs)
            ]

        try:
            if pl_module.training:
                key = "sample_images_train"
            else:
                key = (
                    f"sample_images_val" if not args else f"sample_images_val_{args[0]}"
                )
            pl_module.logger.log_image(
                key=key,
                images=list(xs[:n]) + list(generated[:n]),
                # fuck, why caption instead of captions?
                caption=captions_real[:n] + captions_fake[:n],
                step=pl_module.current_epoch,
            )
            # or use
            # wandb_logger.log_table(key="sample_table", columns=columns, data=data)
        except NotImplementedError:
            pass

    on_train_batch_end = custom_on_batch_end
    on_validation_batch_end = custom_on_batch_end
