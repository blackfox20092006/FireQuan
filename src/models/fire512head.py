from flax import linen as nn
import jax.numpy as jnp
import jax
class Fire(nn.Module):
    squeeze_planes: int
    expand1x1_planes: int
    expand3x3_planes: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=self.squeeze_planes, kernel_size=(1, 1))(x))
        out1x1 = nn.relu(nn.Conv(features=self.expand1x1_planes, kernel_size=(1, 1))(x))
        out3x3 = nn.relu(nn.Conv(features=self.expand3x3_planes, kernel_size=(3, 3), padding='SAME')(x))
        return jnp.concatenate([out1x1, out3x3], axis=-1)
class Fire512(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(8, 32, 32)(x)
        x = Fire(8, 32, 32)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(16, 64, 64)(x)
        x = Fire(16, 64, 64)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(32, 128, 128)(x)
        x = Fire(32, 128, 128)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(features=512, kernel_size=(1, 1))(x)
        x = jnp.mean(x, axis=(1, 2))
        return x
@jax.jit
def cnn_forward(cnn_params, image_batch):
    return Fire512().apply({'params': cnn_params}, image_batch)
