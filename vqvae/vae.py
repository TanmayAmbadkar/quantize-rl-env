import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8x128
        self.conv4 = nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1)  # 6x6xlatent_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # No activation
        
        # return torch.flatten(x, start_dim=1)  # Shape (B, latent_dim, 4, 4)
        return x

class RawObservationEncoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=64):
        super(RawObservationEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x  # Shape (B, latent_dim)
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the embedding vectors
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # x shape: (B, D, H, W)
        # Reshape input to (B * H * W, D)
        x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = x.view(-1, self.embedding_dim)

        # Compute the distances to each embedding
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        # Get the indices of the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)

        # Get the embeddings corresponding to the closest encoding indices
        quantized = self.embeddings(encoding_indices)

        # Reshape the quantized embeddings back to the input shape
        quantized = quantized.view(x.shape)

        # Compute the commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(x, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + commitment_loss


        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Reshape quantized back to (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(x.shape[0], -1)

        return quantized, loss, encoding_indices


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ImageDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            latent_dim, 128, kernel_size=4, stride=2, padding=1
        )  # Upsample to 12x12
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # Upsample to 24x24
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # Upsample to 48x48
        self.deconv4 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # Upsample to 96x96

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Assuming images are normalized between 0 and 1
        return x  # Shape: (B, 3, 96, 96)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, latent_dim=256, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)
        self.image_decoder = ImageDecoder(latent_dim=latent_dim)

        # Vector Quantizer
        self.embedding_dim = latent_dim
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.latent_dim = latent_dim
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, image):
        # Encode images
        image_embedding = self.image_encoder(image)  # Shape: (B, latent_dim, 6, 6)

        # Quantize the embeddings
        quantized_embedding, quantization_loss, encoding_indices = self.quantizer(
            image_embedding
        )  # Quantized shape: (B, latent_dim, 6, 6)

        # Reconstruct images from quantized embeddings
        reconstructed_image = self.image_decoder(quantized_embedding)  # Shape: (B, 3, 96, 96)

        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_image, image)

        # Total loss
        loss = reconstruction_loss + quantization_loss

        return reconstructed_image, loss
    
    def quantize(self, image):
        
        image_embedding = self.image_encoder(image)  # Shape: (B, latent_dim, 6, 6)

        # Quantize the embeddings
        _, _, encoding_indices = self.quantizer(
            image_embedding
        )  # Quantized shape: (B, latent_dim, 6, 6)
        
        return encoding_indices/127
    
    @torch.no_grad()
    def get_quantized_embedding_with_id(self, image):
        
        image_embedding = self.image_encoder(image)  # Shape: (B, latent_dim, 6, 6)

        # Quantize the embeddings
        quantized_embedding, _, encoding_indices = self.quantizer(
            image_embedding
        )  # Quantized shape: (B, latent_dim, 6, 6)

        

        return quantized_embedding, encoding_indices

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
            
            
    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True