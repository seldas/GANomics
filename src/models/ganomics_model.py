import torch
import torch.nn as nn
import itertools
from src.layers.gan_layers import FullyConnectedGenerator, FullyConnectedDiscriminator, GANLoss
from src.layers.init import setup_net

class GANomicsModel:
    """
    GANomics: A generative framework for bidirectional translation between 
    microarray and RNA-seq data using paired-aware feedback loss.
    """
    def __init__(self, input_nc, output_nc, lr=0.0002, betas=(0.5, 0.999), 
                 lambda_A=10.0, lambda_B=10.0, lambda_feedback=10.0, 
                 lambda_idt=0.5, gan_mode='lsgan', device='cpu'):
        
        self.device = device
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_feedback = lambda_feedback
        self.lambda_idt = lambda_idt
        
        # Define Networks
        # G_A: A -> B (Microarray -> RNA-seq)
        # G_B: B -> A (RNA-seq -> Microarray)
        self.netG_A = FullyConnectedGenerator(input_nc, output_nc)
        self.netG_B = FullyConnectedGenerator(output_nc, input_nc)
        
        self.netG_A = setup_net(self.netG_A, device=device)
        self.netG_B = setup_net(self.netG_B, device=device)
        
        # Discriminators
        # D_A: G_A(A) vs B
        # D_B: G_B(B) vs A
        self.netD_A = FullyConnectedDiscriminator(output_nc)
        self.netD_B = FullyConnectedDiscriminator(input_nc)
        
        self.netD_A = setup_net(self.netD_A, device=device)
        self.netD_B = setup_net(self.netD_B, device=device)
        
        # Losses
        self.criterionGAN = GANLoss(gan_mode).to(device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.criterionFeedback = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=lr, betas=betas
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr, betas=betas
        )
        
        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'feedback_A', 'feedback_B', 'idt_A', 'idt_B']
        self.losses = {}

    def set_input(self, input):
        """
        input dict contains:
        'A': source sample
        'B': target sample (unpaired)
        'medianB': paired target for A
        'medianA': paired source for B
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.paired_B = input['medianB'].to(self.device) # paired counterpart for A
        self.paired_A = input['medianA'].to(self.device) # paired counterpart for B
        
        # Ensure correct dimensions for 1x1 convolutions (batch, channels, 1, 1)
        if self.real_A.dim() == 2:
            self.real_A = self.real_A.unsqueeze(-1).unsqueeze(-1)
            self.real_B = self.real_B.unsqueeze(-1).unsqueeze(-1)
            self.paired_B = self.paired_B.unsqueeze(-1).unsqueeze(-1)
            self.paired_A = self.paired_A.unsqueeze(-1).unsqueeze(-1)

    def forward(self):
        # Forward cycles
        self.fake_B = self.netG_A(self.real_A)  # G_A(A) -> B'
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A)) -> A''
        
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) -> A'
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B)) -> B''

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.losses['D_A'] = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.losses['D_B'] = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        # Identity loss (optional)
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.losses['idt_A'] = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.losses['idt_B'] = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.losses['idt_A'] = 0
            self.losses['idt_B'] = 0

        # GAN loss
        self.losses['G_A'] = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.losses['G_B'] = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Cycle loss
        self.losses['cycle_A'] = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        self.losses['cycle_B'] = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
        
        # Feedback (Paired) loss - the GANomics innovation
        self.losses['feedback_B'] = self.criterionFeedback(self.fake_B, self.paired_B) * self.lambda_feedback
        self.losses['feedback_A'] = self.criterionFeedback(self.fake_A, self.paired_A) * self.lambda_feedback
        
        # Combined loss
        loss_G = sum(self.losses[name] for name in ['G_A', 'G_B', 'cycle_A', 'cycle_B', 'feedback_A', 'feedback_B', 'idt_A', 'idt_B'])
        loss_G.backward()
        self.losses['G_total'] = loss_G

    def optimize_parameters(self):
        self.forward()
        
        # Update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # Update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        return {k: v.item() if torch.is_tensor(v) else v for k, v in self.losses.items()}

    def save_networks(self, save_path):
        torch.save({
            'G_A': self.netG_A.state_dict(),
            'G_B': self.netG_B.state_dict(),
            'D_A': self.netD_A.state_dict(),
            'D_B': self.netD_B.state_dict(),
        }, save_path)

    def load_networks(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.netG_A.load_state_dict(checkpoint['G_A'])
        self.netG_B.load_state_dict(checkpoint['G_B'])
        self.netD_A.load_state_dict(checkpoint['D_A'])
        self.netD_B.load_state_dict(checkpoint['D_B'])
