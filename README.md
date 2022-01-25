# FACT: Explaining in Style - Reproducibility Study

## Notes on factors of variation during training

  Quotes are words of author.

### Optimization procedure


- Alternating training - Check influence on training time, image sharpness and stability 

	> We noticed this creates sharper looking images than just training an encoder-decoder network, however it's possible to use only an encoder, and the method would still work.

	Our results: Alternating seems to help on FFHQ according to Noah. Inconclusive results for other datasets, most likely due to other optimization problems.
	
	TODO: Check influence on a simple dataset like MNIST. Perhaps compare on 64x64 on plants/faces once we've got other hyperparameters down
	
	Note: The functional mapping transforms noise to **W**, while the encoder transforms an image. Could there be trouble in training a GAN using **W**  s coming from different networks, thus possibly having different distributions?
	
	Note2: I have a feeling the PL regularization may serve to connect these two distributions, as it keeps track of **W** statistics and make sure steps from the mean don't cause a lot of artifacts. Perhaps something to check.
	
- Updating encoder-generator autoencoder using reconstruction and KL loss during **discriminator** training
	> *Waiting on author's response.*
	
	Our results: Definitely makes MNIST train faster to legible images. Training on plants doesn't seem to suffer (need a good comparison with a baseline).
	
	Note: Usually, the generator is fixed while the discriminator is updated, so at face-value it seems questionable to nudge the generator at this time. However, it may be reasonable to do so as the rec/KL loss is a different one from the adversarial loss. Rec/KL loss should work to make the generator better, without making it better at fooling the discriminator.
	
 - Encoder learning rate - Check influence on stability and training time
	> *Waiting on author's response.*
	
	Background: The functional mapping which takes **Z** to **W** in the original StyleGAN has a learning rate which is 0.01 times the base learning rate. Since our encoder serves to produce the **W** as well, it might be fruitful to make it's learning rate smaller as well. 
	
	Our results: Training on plant dataset with the default learning rate led to unstable training (large, frequent reconstruction loss spikes), with no long-term decrease in reconstruction loss. Decreasing encoder learning rate by a factor of 0.05 produced a stabler training, with the reconstruction loss actually going down. This is with 
	```KL+Rec during discriminator training=True```. 
	
### Hyperparams of interest

- lr - How does the sweet spot change depending on image resolution and dataset? Any interactions with the abovementioned optimization procedure?
	> We used a learning rate of 0.002
	
	Note: Their default learning rate is 10 times bigger than ours. Perhaps we can up the learning rate, assuming the encoder learning rate is lowered accordingly?
- Loss scaling
	1. Reconstruction loss
		>	image (using LPIPS) and on the W-vector (using L1 loss), both with weight 0.1 **+ L1 on images using weight 1**
	
		Results: Higher reconstruction loss scalings seem to make for faster training in our case. We generally scaled by 5 or 10.
		Note: 5/10 is of the order of the difference in learning rate. Perhaps we can afford a higher learning rate, resulting in exactly their scaling of the losses?

	2. KL loss
		> Waiting on author's response.
	
		Results: No scaling (1) seems to work okay in our practice, with our default learning rate.

		Best scaling dependent on dataset?
		> Waiting on author's response

- ttur_mult - Flat scaling to discriminator loss (default 1.5)
		Note: Might be worth fiddling with. Google search gives value 1 might be good for FFHQ

* and many more!