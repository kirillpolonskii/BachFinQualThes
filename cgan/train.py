import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cgan import utils
from cgan.model import Generator, Discriminator, initialize_weights
from cgan.utils import load_checkpoint, save_checkpoint

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
LEARNING_RATE = 0.0007
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_CLASSES = 30
GEN_EMBEDDING = 140
Z_DIM = 140
NUM_EPOCHS = 20
FEATURES_CRITIC = 48
FEATURES_GEN = 48
CRITIC_ITERATIONS = 4
LAMBDA_GP = 10
LOAD_CHECKPOINT = False
GRID_SIZE = 4

# Define the transformation to be applied to the data
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
train_set = datasets.ImageFolder('C:/MAI/DIPL/dataset_on_load_old2expanded', transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
if LOAD_CHECKPOINT:
    load_checkpoint(
        torch.load("C:/MAI/DIPL/BQTMapGeneration/results_18/cgan_map_generation.pth.tar"), gen, critic)
else:
    initialize_weights(gen)
    initialize_weights(critic)

# Initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# Tensorboard plotting
fixed_noise = torch.randn(GRID_SIZE, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"C:/MAI/DIPL/BQTMapGeneration/results_18/logs/CGAN_Map_gen/real")
writer_fake = SummaryWriter(f"C:/MAI/DIPL/BQTMapGeneration/results_18/logs/CGAN_Map_gen/fake")
step = 0
best_loss_cr = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(tqdm(train_loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # Equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = utils.gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 10 == 0 and batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            if loss_critic < best_loss_cr or batch_idx % 150 == 0:
                best_loss_cr = loss_critic
                checkpoint = {
                    'gen': gen.state_dict(), 'disc': critic.state_dict()
                }
                save_checkpoint(checkpoint)

            with torch.no_grad():
                fake = gen(noise, labels)
                img_grid_real = torchvision.utils.make_grid(real[:GRID_SIZE], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:GRID_SIZE], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_real.add_scalar("Discriminator loss", loss_critic, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_fake.add_scalar("Generator loss", loss_gen, global_step=step)

            step += 1

checkpoint = {
    'gen': gen.state_dict(), 'disc': critic.state_dict()
}
save_checkpoint(checkpoint)
