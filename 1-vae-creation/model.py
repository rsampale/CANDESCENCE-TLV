import torch
from torch import nn
import torch.nn.functional as F
import logging

### Logging Info ###

logging.basicConfig(filename='training.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

####################

class Encoder(nn.Module):
    def __init__(self, arguments, OH_len):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, arguments['kernel_size'], stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, arguments['kernel_size'], stride=1, padding=1) 
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, arguments['kernel_size'], stride=2, padding=1) 

        dummy_input = torch.ones(1, 1, 135, 135) 
        dummy_output = self.forward_conv(dummy_input)
        conv_output_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]


        self.linear1 = nn.Linear(conv_output_size + OH_len, arguments['intermediate_dim']) # ADDING OH_len ALLOWS FOR CHANGES TO OH ENCODINGS
        self.linear2 = nn.Linear(arguments['intermediate_dim'], arguments['latent_dim']) # for mu
        self.linear3 = nn.Linear(arguments['intermediate_dim'], arguments['latent_dim']) # for sigma

        self.device = arguments['DEVICE']

        self.N = torch.distributions.Normal(0, 1) # VAEs try to fit to a normal distrubution
        self.N.loc = self.N.loc.to(self.device) # GPU sampling hack
        self.N.scale = self.N.scale.to(self.device)

        self.sigma = 0
        self.mu = 0

    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.batch2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x, one_hot): # Assemble the layers defined above
        logger.info(f"Encoder input shape: {x.shape}")
        logger.info(f"Encoder input has NaN: {torch.isnan(x).any()}")

        x = x.to(self.device)
        x = self.forward_conv(x)

        logger.info(f"After encoder forward conv part shape: {x.shape}")
        logger.info(f"After encoder forward conv part has NaN: {torch.isnan(x).any()}")

        x = torch.flatten(x, start_dim=1)

        # CONCATENATE / ADD ONE-HOT ENCODING OF PLATE HERE
        one_hot = one_hot.squeeze(1)
        one_hot = one_hot.float()
        logger.info(f"Shape of x, before concat attempt: {x.shape}")
        logger.info(f"NaNs in x, before concat attempt: {torch.isnan(x).any()}")
        logger.info(f"One hot shape, before concat attempt: {one_hot.shape}")
        logger.debug(f"One hot size to add in architecture: {one_hot.shape[1]}")
        
        x = torch.cat((x, one_hot), dim=1)


        x = F.relu(self.linear1(x))
        logger.info(f"x shape after linear1: : {x.shape}")
        logger.info(f"after linear1 has NaN: {torch.isnan(x).any()}")
        logger.info(f"after linear1 min/mean/max: {x.min()}, {x.mean()}, {x.max()}")

        mu = self.linear2(x)
        logger.info(f"mu shape: : {mu.shape}")
        logger.info(f"mu has NaN: {torch.isnan(mu).any()}")
        logger.info(f"mu min/mean/max: {mu.min()}, {mu.mean()}, {mu.max()}")
        # print("min/mean/max OUTPUT OF LINEAR3 BEFORE torch.exp: ", self.linear3(x).min().item(), self.linear3(x).mean().item(), self.linear3(x).max().item())

        sigma = torch.exp(self.linear3(x))
        logger.info(f"sigma shape: : {sigma.shape}")
        logger.info(f"sigma has NaN: {torch.isnan(sigma).any()}")
        logger.info(f"sigma min/mean/max: {sigma.min()}, {sigma.mean()}, {sigma.max()}")

        z = mu + sigma*self.N.sample(mu.shape) # reparameterization trick
        logger.info(f"sigma shape: : {sigma.shape}")
        logger.info(f"sigma has NaN: {torch.isnan(sigma).any()}")
        logger.info(f"sigma min/mean/max: {sigma.min()}, {sigma.mean()}, {sigma.max()}")

        self.sigma = sigma
        self.mu = mu
        return z
    
    def encode(self, x, one_hot):
        self.forward(x, one_hot)
        return self.mu
    

class Decoder(nn.Module):
    def __init__(self, arguments, OH_len, OH_in_decoder):
        super().__init__()

        self.include_OH = OH_in_decoder
        if OH_in_decoder:
            self.linear_part = nn.Sequential(
                nn.Linear(arguments['latent_dim'] + OH_len, arguments['intermediate_dim']), # ADDING OH_len ALLOWS FOR CHANGES TO OH ENCODINGS
                nn.ReLU(True),
                nn.Linear(arguments['intermediate_dim'], 128 * 68 * 68), # possibly change, not perfect because 135 not divisible by 2
                nn.ReLU(True)
            )
        else:
            self.linear_part = nn.Sequential(
                nn.Linear(arguments['latent_dim'], arguments['intermediate_dim']), # Don't allocate space for OH encodings if we don't want it in the decoder
                nn.ReLU(True),
                nn.Linear(arguments['intermediate_dim'], 128 * 68 * 68), # possibly change, not perfect because 135 not divisible by 2
                nn.ReLU(True)
            )

        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = ( 128, 68, 68))

        self.conv_part = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding=1, output_padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride = 1, padding = 1, output_padding = 0), # changed output padding to 0 for error.. double check
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride = 1, padding = 1, output_padding = 0)
        )

    def forward(self, x, one_hot):
        logger.info(f"Decoder input shape: {x.shape}")
        logger.info(f"Decoder input has NaN: {torch.isnan(x).any()}")
        logger.info(f"Decoder input min/mean/max: {x.min()}, {x.mean()}, {x.max()}")
        
        if self.include_OH:
            one_hot = one_hot.squeeze(1)
            one_hot = one_hot.float()
            x = torch.cat((x, one_hot), dim=1)

        x = self.linear_part(x)
        logger.info(f"decoder after linear part shape: {x.shape}")
        logger.info(f"decoder after linear part has NaN: {torch.isnan(x).any()}")
        logger.info(f"decoder after linear part min/mean/max: {x.min()}, {x.mean()}, {x.max()}")

        x = self.unflatten(x)
        logger.info(f"decoder after unflatten shape: {x.shape}")
        logger.info(f"decoder after unflatten has NaN: {torch.isnan(x).any()}")

        x = self.conv_part(x)
        logger.info(f"decoder after conv part shape: {x.shape}")
        logger.info(f"decoder after conv part has NaN: {torch.isnan(x).any()}")

        x = torch.sigmoid(x)
        logger.info(f"Decoder final output shape: {x.shape}")
        logger.info(f"Decoder final output has NaN: {torch.isnan(x).any()}")
        
        return x
    

class VAE(nn.Module):
    def __init__(self, arguments, OH_len, OH_in_decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder(arguments, OH_len)  #change __init__ of encoder/decoder to un-hardcode the OH addition, it is called here
        self.decoder = Decoder(arguments, OH_len, OH_in_decoder)
        self.device = arguments['DEVICE']

    def forward(self, x, one_hot):
        x = x.to(self.device)
        one_hot = one_hot.to(self.device)
        # logger.info(f"\nTENSOR INPUT TO SIZE GETTER FUNCTION: {x.shape}")
        logger.info(f"\nONE-HOT DURING VAE FORWARD PASS: {one_hot.shape}")
        z = self.encoder(x, one_hot)
        return self.decoder(z, one_hot)