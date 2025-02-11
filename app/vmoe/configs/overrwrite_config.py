
def overwrite_config(config, workdir):
    for i in workdir.split(","):
        key, value = i.split("=")
        if key == "ver":
            continue
        
        if key == "base":
            if "full" in workdir:
                print("Warning: base config is only used in fine-tuning. Base settings is ignored.")
                continue
            config.initialization.prefix = value
            if value[:2] != "gs":
                config.initialization.rules = [
                    ('head', ''),              # Do not restore the head params.
                    ('pre_logits/.*', ''),     # Do not restore the pre_logits params.
                    ('opt_state/.*', ''),     # only local ?????????????
                    ('rngs/mixup', ''),     # only local ?????????????
                    ('step', ''),     # only local ?????????????
                    ('^(.*/pos_embedding)$', r'\1', 'vit_zoom'), # only local
                    ('^(.*)$', r'\1'), # only local
                ]
#        if key == "bs":

        if key == "":
