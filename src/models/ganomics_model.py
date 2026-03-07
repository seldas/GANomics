import torch
import torch.nn as nn
import itertools
from src.layers.gan_layers import FullyConnectedGenerator, FullyConnectedDiscriminator, GANLoss
from src.layers.init import setup_net

class GANomicsModel:
    """
    GANomics: A generative framework for bidirectional translation between 
    microarray and RNA-seq data using paired-aware feedback loss.
    Supports 'bidirectional' (default) and 'one_way' modes.
    """
    def __init__(self, input_nc, output_nc, lr=0.0002, betas=(0.5, 0.999), 
                 lambda_A=10.0, lambda_B=10.0, lambda_feedback=10.0, 
                 lambda_idt=0.5, gan_mode='lsgan', device='cpu', direction='both'):
        
        self.device = device
        self.direction = direction # 'both', 'AtoB', or 'BtoA'
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_feedback = lambda_feedback
        self.lambda_idt = lambda_idt
        
        # Define Networks
        self.netG_A = None
        self.netG_B = None
        self.netD_A = None
        self.netD_B = None

        if direction in ['both', 'AtoB']:
            self.netG_A = setup_net(FullyConnectedGenerator(input_nc, output_nc), device=device)
            self.netD_A = setup_net(FullyConnectedDiscriminator(output_nc), device=device)
        
        if direction in ['both', 'BtoA']:
            self.netG_B = setup_net(FullyConnectedGenerator(output_nc, input_nc), device=device)
            self.netD_B = setup_net(FullyConnectedDiscriminator(input_nc), device=device)
        
        # Losses
        self.criterionGAN = GANLoss(gan_mode).to(device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.criterionFeedback = nn.L1Loss()
        
        # Optimizers
        g_params = []
        d_params = []
        if self.netG_A: g_params += list(self.netG_A.parameters())
        if self.netG_B: g_params += list(self.netG_B.parameters())
        if self.netD_A: d_params += list(self.netD_A.parameters())
        if self.netD_B: d_params += list(self.netD_B.parameters())

        self.optimizer_G = torch.optim.Adam(g_params, lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(d_params, lr=lr, betas=betas)
        
        self.loss_names = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'feedback_A', 'feedback_B', 'idt_A', 'idt_B']
        self.losses = {name: 0.0 for name in self.loss_names}

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.paired_B = input['medianB'].to(self.device) 
        self.paired_A = input['medianA'].to(self.device) 
        
        if self.real_A.dim() == 2:
            self.real_A = self.real_A.unsqueeze(-1).unsqueeze(-1)
            self.real_B = self.real_B.unsqueeze(-1).unsqueeze(-1)
            self.paired_B = self.paired_B.unsqueeze(-1).unsqueeze(-1)
            self.paired_A = self.paired_A.unsqueeze(-1).unsqueeze(-1)

    def forward(self):
        if self.direction in ['both', 'AtoB']:
            self.fake_B = self.netG_A(self.real_A)
            if self.direction == 'both':
                self.rec_A = self.netG_B(self.fake_B)
        
        if self.direction in ['both', 'BtoA']:
            self.fake_A = self.netG_B(self.real_B)
            if self.direction == 'both':
                self.rec_B = self.netG_A(self.fake_A)

    def backward_D(self):
        if self.netD_A:
            self.losses['D_A'] = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        if self.netD_B:
            self.losses['D_B'] = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        loss_G = 0
        
        # 1. AtoB Direction
        if self.direction in ['both', 'AtoB']:
            # GAN & Feedback
            self.losses['G_A'] = self.criterionGAN(self.netD_A(self.fake_B), True)
            self.losses['feedback_B'] = self.criterionFeedback(self.fake_B, self.paired_B) * self.lambda_feedback
            loss_G += self.losses['G_A'] + self.losses['feedback_B']
            
            # Cycle & Identity (Only in bidirectional mode)
            if self.direction == 'both':
                self.losses['cycle_A'] = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
                loss_G += self.losses['cycle_A']
                if self.lambda_idt > 0:
                    self.losses['idt_A'] = self.criterionIdt(self.netG_A(self.real_B), self.real_B) * self.lambda_B * self.lambda_idt
                    loss_G += self.losses['idt_A']

        # 2. BtoA Direction
        if self.direction in ['both', 'BtoA']:
            # GAN & Feedback
            self.losses['G_B'] = self.criterionGAN(self.netD_B(self.fake_A), True)
            self.losses['feedback_A'] = self.criterionFeedback(self.fake_A, self.paired_A) * self.lambda_feedback
            loss_G += self.losses['G_B'] + self.losses['feedback_A']
            
            # Cycle & Identity (Only in bidirectional mode)
            if self.direction == 'both':
                self.losses['cycle_B'] = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
                loss_G += self.losses['cycle_B']
                if self.lambda_idt > 0:
                    self.losses['idt_B'] = self.criterionIdt(self.netG_B(self.real_A), self.real_A) * self.lambda_A * self.lambda_idt
                    loss_G += self.losses['idt_B']

        loss_G.backward()
        self.losses['G_total'] = loss_G

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([n for n in [self.netD_A, self.netD_B] if n], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([n for n in [self.netD_A, self.netD_B] if n], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def save_networks(self, save_path):
        state = {}
        if self.netG_A: state['G_A'] = self.netG_A.state_dict()
        if self.netG_B: state['G_B'] = self.netG_B.state_dict()
        if self.netD_A: state['D_A'] = self.netD_A.state_dict()
        if self.netD_B: state['D_B'] = self.netD_B.state_dict()
        torch.save(state, save_path)

    def load_networks(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        if self.netG_A and 'G_A' in checkpoint: self.netG_A.load_state_dict(checkpoint['G_A'])
        if self.netG_B and 'G_B' in checkpoint: self.netG_B.load_state_dict(checkpoint['G_B'])
        if self.netD_A and 'D_A' in checkpoint: self.netD_A.load_state_dict(checkpoint['D_A'])
        if self.netD_B and 'D_B' in checkpoint: self.netD_B.load_state_dict(checkpoint['D_B'])
