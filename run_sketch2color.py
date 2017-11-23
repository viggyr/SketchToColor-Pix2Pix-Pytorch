#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:21:32 2017

@author: vignesh
"""
import os.path
import image_loader
import visdom
import model
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

vis=visdom.Visdom()
discriminator_path = '/home/vignesh/miniproject/discriminator.pt'
generator_path = '/home/vignesh/miniproject/generator.pt'
def convert_to_rgb(one):
    one=one+1;
    three=one*127
    return three

def run():
    edges_batch = torch.FloatTensor(4, 1, 256, 256)
    color_batch = torch.FloatTensor(4, 3, 256, 256)
    ones_label = torch.ones(4)
    zeros_label = torch.zeros(4)
    
    
    discriminator=model.Discriminator_Model()
    if os.path.isfile(discriminator_path):
        print "reusing discriminator model parameters"
        discriminator.load_state_dict(torch.load(discriminator_path))
    else:
        print "created discriminator model with fresh parameters"
        discriminator.apply(model.weights_init)
    
    generator=model.Generator_model()
    generator.apply(model.weights_init)
    if os.path.isfile(generator_path):
        print "reusing generator model parameters"
        generator.load_state_dict(torch.load(generator_path))
    else:
        print "created generator model with fresh parameters"
    discriminator=discriminator.cuda()
    generator=generator.cuda()
    edges_batch=edges_batch.cuda()
    color_batch=color_batch.cuda()
    ones_label=ones_label.cuda()
    zeros_label = zeros_label.cuda()
    
    edges_batch = Variable(edges_batch)
    color_batch = Variable(color_batch)
    ones_label = Variable(ones_label)
    zeros_label = Variable(zeros_label)
    
    D_solver = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    G_solver = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    losses=model.losses();
    training_loader=image_loader.batches()
    
    max_iter=15
    
    for iter in range(0,max_iter):     
        print "iteration {}".format(iter)
        for batch,(edge,color) in enumerate(training_loader):
            discriminator.zero_grad()
            edges_batch.data.resize_(edge.size()).copy_(edge)
            color_batch.data.resize_(color.size()).copy_(color)
            D_real=discriminator(edges_batch,color_batch)
            D_x_y = D_real.data.mean()
            
            G_fake = generator(edges_batch)
            D_fake = discriminator(edges_batch, G_fake.detach())
            D_x_gx = D_fake.data.mean()
            D_loss = losses.dloss(D_real,D_fake)
            D_loss.backward()
            D_solver.step()
            
            generator.zero_grad()
            G_fake = generator(edges_batch)
              #vis.images(G_fake.transpose())
            #writer.add_image('Generated Images ',torch.cat([edges_batch.data.permute(0,2,3,1),G_fake.data.permute(0,2,3,1)],2)
            D_fake = discriminator(edges_batch,G_fake)
            D_x_y = D_fake.data.mean()
            G_loss=losses.gloss(G_fake,D_fake,color_batch)
            G_loss.backward()
            G_solver.step()
            if (batch+1)%5 == 0:
                vis.images(torch.cat([torch.cat([convert_to_rgb(edges_batch.data),convert_to_rgb(edges_batch.data),convert_to_rgb(edges_batch.data)],1),G_fake.data],3).cpu().numpy(),opts=dict(title='Images', caption='iter {}'.format(iter)),)
                print "batch {} D loss : {} G loss: {} D_x D_gx ".format(batch,D_loss.data[0],G_loss.data[0],D_x_y,D_x_gx) 
                torch.save(discriminator.state_dict(), discriminator_path)
                torch.save(generator.state_dict(), generator_path)
        

def main():
    run()
if __name__ == '__main__':
    main()    
