from model.modules.latent.parameters_inference import ParametersInference
from model.modules.recurrent.seq_to_seq import SeqToSeq
from model.modules.recurrent.seq_to_vec import SeqToVec
from model.modules.recurrent.vec_to_seq import VecToSeq
from .abstract_vae.generative import GenerativeBlock
from .abstract_vae.inference import InferenceBlock
from .conv.resnet import ResNet
from .conv.seq_resnet import SeqResNet
from .embedding import Embedding
from .latent.iaf.highway import Highway
from .latent.iaf.iaf import IAF
from .latent.posterior_combination import PosteriorCombination
from .utils.view import View
