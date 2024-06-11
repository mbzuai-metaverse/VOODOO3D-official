from utils.registry import MODEL_REGISTRY

# Register additional modules
import additional_modules.gfpgan.gfpganv1_clean_arch



def get_model(opt):

    MODEL_REGISTRY.scan_and_register()
    return MODEL_REGISTRY.get(opt['model_class'])(**opt['params'])
