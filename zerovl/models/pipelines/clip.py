import numpy as np

import torch
import torch.nn as nn

from zerovl.models.pipelines.builder import PIPELINE
from zerovl.models.backbones.builder import BACKBONE
from zerovl.models.criteria.losses.builder import LOSS

from zerovl.utils import logger, ENV
from ..components import SimpleProjection, ComplexProjection, L2norm

class CLIPModel(nn.Module):
    def __init__(self, cfg, rank):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.random_seed = np.random.RandomState(seed=2021)

        if cfg.model.projection.name == "simple":
            ProjectionHead = SimpleProjection
        elif cfg.model.projection.name == "complex":
            ProjectionHead = ComplexProjection
        else:
            raise NotImplementedError

        self.image_projection = ProjectionHead(
            cfg, embedding_dim=cfg.model.image_encoder.embedding_dim, projection_dim=cfg.model.projection.dim, trainable=cfg.model.projection.image_projector_trainable
            )
        self.text_projection = ProjectionHead(
            cfg, embedding_dim=cfg.model.text_encoder.embedding_dim, projection_dim=cfg.model.projection.dim, trainable=cfg.model.projection.text_projector_trainable
            )

        self.text_pool = nn.Identity()
        self.image_pool = nn.Identity()

        self.loss = LOSS.get(cfg.loss.name)(cfg, rank)
        
        self.extra_loss_names = cfg.loss.extra_losses
        self.extra_loss = nn.ModuleList()
        if len(cfg.loss.extra_losses) > 0:
            logger.info(f'Using extra losses {cfg.loss.extra_losses}')
            for loss_name in cfg.loss.extra_losses:
                self.extra_loss.append(LOSS.get(loss_name)(cfg, rank))

        self.global_reduce = cfg.loss.global_reduce
        self.text_target_token_idx = cfg.model.text_encoder.target_token_idx

        # Define and check mixup params 
        self.use_mixup = 'MixUp' in self.cfg.loss.name


    def train(self, mode=True):
        # Override train so that the training mode is set as we want (BN does not update the running stats)
        nn.Module.train(self, mode)
        if mode and self.cfg.model.freeze_cnn_bn:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            print("freezing bn in image encoder")
            self.image_encoder.apply(set_bn_eval)

    def get_mixup_kwargs(self, mixup_kwargs):
        # Sample [alpha] for mixup strategy and [gamma] for coin flipping mixup
        alpha = self.random_seed.beta(self.cfg.loss.mixup.beta, self.cfg.loss.mixup.beta)
        image_alpha, text_alpha = alpha, alpha
        gamma = self.random_seed.random() 

        # if gamma > 0.5, carrying out only image mixup
        # elif gamma <= 0.5, carrying out only text mixup
        if gamma > 0.5:
            mixup_kwargs['image_alpha'] = image_alpha
        else:
            mixup_kwargs['text_alpha'] = text_alpha
        return mixup_kwargs

    def forward_image_feature(self, image, **kwargs):

        # raw image mixup
        if self.use_mixup and 'image_alpha' in kwargs:
            flip_image = image.flip(0)
            alpha = kwargs['image_alpha'] 
            image = alpha * image + (1-alpha) * flip_image

        image_features = self.image_encoder(image, **kwargs)
        
        # check whether use ViT with only [CLS] token
        if self.cfg.model.image_encoder.vit.only_cls_token and len(image_features.shape) == 3:
            image_features = image_features[:, 0]

        return image_features

    def forward_image_project(self, image_features, **kwargs):
        image_embeddings = self.image_pool(self.image_projection(image_features))

        if self.cfg.model.projection.name == 'simple':
            image_embeddings = L2norm(image_embeddings, dim=-1)

        return image_embeddings

    def forward_text_feature(self, input_ids, attention_mask, norm=True, **kwargs):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        return text_features

    def forward_text_project(self, text_features, attention_mask, **kwargs):
        text_embeddings = self.text_projection(text_features[:, self.text_target_token_idx, :])

        # manifold text mixup
        if self.use_mixup and 'text_alpha' in kwargs:
            text_embeddings = self.text_pool(text_embeddings)
            alpha = kwargs['text_alpha']
            text_embeddings = alpha * text_embeddings + (1-alpha) * text_embeddings.flip(0)
        else:
            text_embeddings = self.text_pool(text_embeddings)

        if self.cfg.model.projection.name == 'simple':
            text_embeddings = L2norm(text_embeddings, dim=-1)

        return text_embeddings


    def forward(self, batch, embeddings=False, **kwargs):
        if self.cfg.runner.name != 'clip_bsgs':
            mixup_kwargs = {}
            if self.use_mixup and not embeddings:
                mixup_kwargs = self.get_mixup_kwargs(mixup_kwargs)

            image_embeddings = self.forward_image_feature(batch["image"], **mixup_kwargs)
            text_embeddings = self.forward_text_feature(batch["input_ids"], batch["attention_mask"], **mixup_kwargs)

            image_embeddings = self.forward_image_project(image_embeddings, **mixup_kwargs)
            text_embeddings = self.forward_text_project(text_embeddings, batch["attention_mask"], **mixup_kwargs)

            if embeddings == "all":
                return [image_embeddings, text_embeddings]

            # Calculating the Loss
            if self.global_reduce:
                i2t_loss, i2t_acc = self.loss(
                    image_embeddings,
                    text_embeddings,
                    ignore_mask=None,
                    **mixup_kwargs
                )
                t2i_loss, t2i_acc = self.loss(
                    text_embeddings,
                    image_embeddings,
                    ignore_mask=None,
                    **mixup_kwargs
                )
                loss = 0.5 * (i2t_loss + t2i_loss)
            else:
                loss, i2t_acc, t2i_acc = self.loss(
                    image_embeddings, text_embeddings, ignore_mask=None
                )
            
            loss_dict = {}
            loss_dict[f'{self.cfg.loss.name}_loss'.lower()] = loss
            for loss_func, loss_name in zip(self.extra_loss, self.extra_loss_names):
                loss = loss_func(image_embeddings, text_embeddings, batch["dataset_index"], ignore_mask=None)
                loss_dict[f'{loss_name}_loss'.lower()] = loss
            return loss_dict, i2t_acc, t2i_acc

        else:
            mixup_kwargs = kwargs

            image_embeddings = self.forward_image_feature(batch["image"], **mixup_kwargs)
            text_embeddings = self.forward_text_feature(batch["input_ids"], batch["attention_mask"], **mixup_kwargs)

            image_embeddings = self.forward_image_project(image_embeddings, **mixup_kwargs)
            text_embeddings = self.forward_text_project(text_embeddings, batch["attention_mask"], **mixup_kwargs)

            temp = torch.clamp(self.loss.temperature, 0.001, 0.5)
            
            return image_embeddings, text_embeddings, temp


class ImageEncoder(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_tag = cfg.model.image_encoder.tag
        self.pretrained = cfg.model.image_encoder.pretrained
        self.trainable = cfg.model.image_encoder.trainable

        kwargs_dict = {}
        if "vit" not in self.model_tag: # enable/disable the global average pooling for CNN.
            if cfg.model.pool.name in ["gpo"]:
                kwargs_dict["global_pool"] = ""
            else:
                kwargs_dict["global_pool"] = "avg"
        if "vit" in self.model_tag: # specify the input_size for intializing ViTs with timm.
            kwargs_dict['img_size'] = cfg.transforms.input_size
        
        model_builder = BACKBONE.get(cfg.model.image_encoder.name)
        self.model = model_builder(cfg, **kwargs_dict)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x, **kwargs):
        x = self.model(x, **kwargs)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_tag = cfg.model.text_encoder.tag
        self.pretrained = cfg.model.text_encoder.pretrained
        self.trainable = cfg.model.text_encoder.trainable

        model_builder = BACKBONE.get(cfg.model.text_encoder.name)
        self.model = model_builder(cfg)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state


@PIPELINE.register_obj
def clip(cfg):
    model = CLIPModel(cfg, ENV.rank)
    return model
