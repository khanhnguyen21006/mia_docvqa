def build_model(model_name, config):

    available_models = ['vt5']

    if model_name.lower() == 'vt5':
        from modules.pfl.models.VT5 import VT5
        model = VT5(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(model_name, ', '.join(available_models)))

    model.model.to(config['device'])
    return model


def build_dataset(model_name, dataset_name, split, gt_dir, imgs_dir, client_id=None):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if client_id is not None:
        dataset_kwargs['client_id'] = client_id

    # Build dataset
    if dataset_name == 'PFL-DocVQA':
        from modules.pfl.datasets.PFL_DocVQA import PFL_DocVQA
        dataset = PFL_DocVQA(gt_dir, imgs_dir, split, dataset_kwargs)

    else:
        raise ValueError

    return dataset

