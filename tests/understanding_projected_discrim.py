import sys
sys.path.append('../eg3d')
import dnnlib
from training.pg_modules.discriminator import ProjectedDiscriminator


image_pair_discrim = ProjectedDiscriminator(
            backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'], diffaug=True, interp224=False,
            backbone_kwargs=dnnlib.EasyDict(),)