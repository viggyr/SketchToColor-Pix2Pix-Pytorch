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
discriminator_path = 'discriminator.pt'
generator_path = 'generator.pt'

#rescale from [-1,1] to [0,255]
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
    #Load discriminator parameters if available
    if os.path.isfile(discriminator_path):
        print "reusing discriminator model parameters"
        discriminator.load_state_dict(torch.load(discriminator_path))
    else:
        print "created discriminator model with fresh parameters"
        discriminator.apply(model.weights_init)
    
    generator=model.Generator_model()
    #Load generated parameters if available
        
    if os.path.isfile(generator_path):
        print "reusing generator model parameters"
        generator.load_state_dict(torch.load(generator_path))
    else:
        print "created generator model with fresh parameters"
        generator.apply(model.weights_init)
        
    #send data to gpu
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
            #Train discriminator on real data
            discriminator.zero_grad()
            edges_batch.data.resize_(edge.size()).copy_(edge)
            color_batch.data.resize_(color.size()).copy_(color)
            D_real=discriminator(edges_batch,color_batch)
            
            #Train discriminator on fake(generated) data
            G_fake = generator(edges_batch)
            D_fake = discriminator(edges_batch, G_fake.detach())
            D_loss = losses.dloss(D_real,D_fake)
            D_loss.backward()
            D_solver.step()
            
            #train generatpr
            generator.zero_grad()
            G_fake = generator(edges_batch)
            D_fake = discriminator(edges_batch,G_fake)
            D_x_y = D_fake.data.mean()
            G_loss=losses.gloss(G_fake,D_fake,color_batch)
            G_loss.backward()
            G_solver.step()
            
            #print progress and visualize the images generated
            if (batch+1)%100 == 0:
                #Edge Images
                vis.images(torch.cat([convert_to_rgb(edges_batch.data),convert_to_rgb(edges_batch.data),convert_to_rgb(edges_batch.data)],1).cpu().numpy()[0:4],opts=dict(title='Sketch Images', caption='iter {}'.format(iter)),)
                #Generated Images
                vis.images( convert_to_rgb(G_fake.data).cpu().numpy()[0:4],opts=dict(title='Generated Images', caption='iter {}'.format(iter)),)
                #Actual colored images
                vis.images( convert_to_rgb(color_batch.data).cpu().numpy()[0:4],opts=dict(title='Expected Images', caption='iter {}'.format(iter)),)
              
                print "batch {} D loss : {} G loss: {} ".format(batch,D_loss.data[0],G_loss.data[0]) 
                
                #Save model parameters
                torch.save(discriminator.state_dict(), discriminator_path)
                torch.save(generator.state_dict(), generator_path)
        

def main():
    run()
if __name__ == '__main__':
    main()    
