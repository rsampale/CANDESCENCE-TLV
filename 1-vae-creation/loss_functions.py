import torch
import logging

### Logging Info ###

logging.basicConfig(filename='loss.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

####################

def get_KL_loss(sigma, mu):
    kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return kl_loss

def get_MSE_loss(x, x_hat):
    mse_loss = ((x - x_hat)**2).sum()
    return mse_loss

def get_MSE_kl_loss(x, x_hat, sigma, mu, arguments):
    # Total loss is defined as MSE loss + KL-divergence loss
    kl_weight = arguments['kl_weight']
    MSE_weight = arguments['MSE_weight']
    kl_loss = get_KL_loss(sigma, mu)
    MSE_loss = get_MSE_loss(x, x_hat)
    logging.info(f"MSE LOSS: {MSE_loss}")
    logging.info(f"KL LOSS: {kl_loss}")
    total_loss = (MSE_loss * MSE_weight) + (kl_loss * kl_weight)
    return total_loss