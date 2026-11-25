import clip
import torch
from torch import nn
from torch.nn import functional as F


class FeatureCNN(nn.Module):
    def __init__(self, input_dim, args):
        super(FeatureCNN, self).__init__()

        layers = []
        num_layers = args.featureCNNLayer

        for i in range(num_layers):
            layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(input_dim))

            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.spatial_conv = nn.Sequential(*layers)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.spatial_conv(x)
        out = out + identity
        out = self.relu(out)

        return out


class ClipEncoder(nn.Module):
    mask_layers = (19, 23)

    def __init__(self, path="./RemoteCLIP-ViT-L-14.pt"):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-L/14", device=device, jit=False)

        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

        self.register_buffer(
            "target_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "target_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

        self.features = {}
        for layer in self.mask_layers:
            self.model.visual.transformer.resblocks[layer].register_forward_hook(
                self._capture_features(layer)
            )

    def _capture_features(self, layer):
        def hook(_module, _inputs, output):
            self.features[layer] = output.detach().clone()

        return hook

    def preprocess(self, image):
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        return (image - self.target_mean) / self.target_std

    def _patch_grid(self, tokens):
        tokens = self.model.visual.ln_post(tokens.transpose(0, 1))
        patches = tokens[:, 1:, :]
        batch, num_patches, channels = patches.shape
        grid_size = int(num_patches ** 0.5)
        return patches.reshape(batch, grid_size, grid_size, channels).permute(0, 3, 1, 2)

    @torch.no_grad()
    def forward(self, image):
        self.features.clear()
        self.model.encode_image(self.preprocess(image))
        mid, high = self.mask_layers
        return self._patch_grid(self.features.pop(mid)), self._patch_grid(self.features.pop(high))


class Encoder(nn.Module):
    def __init__(self, network, args):
        super().__init__()
        if network != "remoteClip":
            raise ValueError("Encoder requires --network remoteClip")
        if args.mask != "Self":
            raise ValueError("Encoder requires --mask Self")

        self.model = ClipEncoder()
        self.fine_tune()

        self.featureCNN_mid = (
            FeatureCNN(args.encoder_dim, args) if args.featureCNN == "True" else None
        )

    def fine_tune(self, fine_tune=True):
        # The training script calls this method. RemoteCLIP stays frozen.
        self.model.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older encoder checkpoints include these unused parameters.
        for name in ("mask_layer_weights", "gamma_mid", "gamma_low"):
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, imageA, imageB):
        featA_mid, featA_high = self.model(imageA)
        featB_mid, featB_high = self.model(imageB)

        if self.featureCNN_mid is not None:
            featA_mid = self.featureCNN_mid(featA_mid.float())
            featB_mid = self.featureCNN_mid(featB_mid.float())
            featA_high = self.featureCNN_mid(featA_high.float())
            featB_high = self.featureCNN_mid(featB_high.float())

        mid_mask = (1.0 - F.cosine_similarity(featA_mid, featB_mid, dim=1, eps=1e-8)).unsqueeze(1)
        high_mask = (1.0 - F.cosine_similarity(featA_high, featB_high, dim=1, eps=1e-8)).unsqueeze(1)

        featA = featA_high * high_mask + 0.2 * (featA_mid * mid_mask)
        featB = featB_high * high_mask + 0.2 * (featB_mid * mid_mask)
        return featA, featB, high_mask


class MiddleLayer(nn.Module):

    def __init__(self, feature_size):
        super(MiddleLayer, self).__init__()
        h_feat, w_feat, channels = feature_size

        self.time_embedding = nn.Embedding(2, channels)

        self.time_embedding.weight.data.normal_(0, 0.02)

        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2, mask=None):

        batch, c, h, w = img1.shape

        pos_h = torch.arange(h).to(img1.device)
        pos_w = torch.arange(w).to(img1.device)
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)

        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim = -1)
        pos_embedding = pos_embedding.permute(2,0,1).unsqueeze(0).repeat(batch, 1, 1, 1)
        img1 = img1 + pos_embedding
        img2 = img2 + pos_embedding

        time_emb_before = self.time_embedding(torch.tensor(0).to(img1.device))
        time_emb_after  = self.time_embedding(torch.tensor(1).to(img1.device))

        img1 = img1 + time_emb_before.view(1, c, 1, 1)
        img2 = img2 + time_emb_after.view(1, c, 1, 1)

        img1 = img1.view(batch, c, -1).transpose(-1, 1)
        img2 = img2.view(batch, c, -1).transpose(-1, 1)

        img1 = img1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2
