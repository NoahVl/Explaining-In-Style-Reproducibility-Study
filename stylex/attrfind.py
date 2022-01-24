# For the attribute find functions.
import os

from stylex_train import StylEx, Dataset, DistributedSampler, MNIST_1vA

# For setting data src.
from stylex_train import image_noise, styles_def_to_tensor, make_weights_for_balanced_classes, cycle, default

from mobilenet_classifier import MobileNet
import torch
from torch.utils import data
import math
import tqdm

import multiprocessing
from torchvision.utils import make_grid
from PIL import Image

# For reading lines and converting them to dict.
import ast

NUM_CORES = multiprocessing.cpu_count()


def plot_image(tensor) -> None:
    """
    Plots an image from a tensor.
    """
    grid = make_grid(tensor)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    # Plot PIL image using plt.imshow
    im.show()


def set_data_src(folder='./', dataset_name=None, image_size=32, batch_size=16, num_workers=4,
                 is_ddp=False, rank=0, world_size=1):
    if dataset_name is None:
        dataset = Dataset(folder, image_size)
        num_workers = default(num_workers, NUM_CORES if not is_ddp else 0)

        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size,
                                     shuffle=True) if is_ddp else None

        dataloader = data.DataLoader(dataset, num_workers=num_workers,
                                     batch_size=math.ceil(batch_size / world_size), sampler=sampler,
                                     shuffle=not is_ddp, drop_last=True, pin_memory=True)

    elif dataset_name == 'MNIST':

        dataset = MNIST_1vA(digit=8)

        # weights = make_weights_for_balanced_classes(dataset.dataset, num_classes)
        # sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        dataloader = data.DataLoader(dataset, batch_size=batch_size)
    else:
        raise NotImplementedError("This dataset is not supported yet. Please use dataset_name = None.")

    loader = cycle(dataloader)

    return dataset, loader


def sindex_to_block_idx_and_index(generator, sindex):
    tmp_idx = sindex

    block_idx = None
    idx = None

    for idx, block in enumerate(generator.blocks):
        if tmp_idx < block.num_style_coords:
            block_idx = idx
            idx = tmp_idx
            break
        else:
            tmp_idx = tmp_idx - block.num_style_coords

    return block_idx, idx


def _float_features(values):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def get_min_max_style_vectors(stylex, classifier, loader, batch_size, num_images=264, cuda_rank=0, image_size=32):
    minimums = None
    maximums = None

    minimums = None
    maximums = None

    for _ in range(num_images // batch_size):
        encoder_batch = next(loader).cuda(cuda_rank)

        encoder_output = stylex.encoder(encoder_batch)
        real_classified_logits = classifier.classify_images(encoder_batch)

        # noise = image_noise(batch_size, image_size, device=cuda_rank)
        zero_noise = torch.zeros(batch_size, image_size, image_size, 1).cuda(cuda_rank)

        latent_w = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                     stylex.G.num_layers)]  # Has to be bracketed because expects a noise mix

        # dlatents.append(encoder_output)
        w_latent_tensor = styles_def_to_tensor(latent_w)

        rgb, style_coords = stylex.G(w_latent_tensor, zero_noise, get_style_coords=True)

        if minimums is None or maximums is None:
            minimums = style_coords
            maximums = style_coords
        else:
            minimums = torch.minimum(minimums, style_coords)
            maximums = torch.maximum(maximums, style_coords)

    minimums = torch.min(minimums, dim=0)[0]
    maximums = torch.max(maximums, dim=0)[0]

    return minimums, maximums


def filter_unstable_images(style_change_effect: torch.Tensor,
                           effect_threshold: float = 0.3,
                           num_indices_threshold: int = 150) -> torch.Tensor:
    """Filters out images which are affected by too many S values."""
    unstable_images = (
            torch.sum(torch.abs(style_change_effect) > effect_threshold, dim=(1, 2, 3)) > num_indices_threshold)
    style_change_effect[unstable_images] = 0
    return style_change_effect


def run_attrfind(
        data='./',
        stylex_path='',
        classifier_name='',
        image_size=32,
        num_images=4,
        batch_size=16,
        dataset_name=None,
        s_shift_size=1,
        att_find_text_file="./att_find/att_find_computations.txt"
):
    with torch.no_grad():
        dataset, loader = set_data_src(data, dataset_name, image_size, batch_size)
        # Since attribute find is not helped by having multiple GPU's, we hardcode cuda_rank to 0.
        cuda_rank = 0

        stylex = StylEx(image_size=image_size)

        # smt like this
        stylex.load_state_dict(torch.load(stylex_path)["StylEx"])

        classifier = MobileNet(classifier_name, cuda_rank=cuda_rank, output_size=2, image_size=image_size)

        minimums, maximums = get_min_max_style_vectors(stylex, classifier, loader, batch_size=batch_size,
                                                       cuda_rank=cuda_rank, image_size=image_size)

        # Dictionary of features
        style_vector_amount = len(minimums)

        # Check if the att_find_text_file path exists, otherwise create a directory with the text file in there
        if not os.path.exists(os.path.dirname(att_find_text_file)):
            os.makedirs(os.path.dirname(att_find_text_file))

        zero_noise = torch.zeros(batch_size, image_size, image_size, 1).cuda(cuda_rank)

        # Check if the att_find_text_file exists, otherwise do the attfind computations.
        if not os.path.exists(att_find_text_file):
            with open(att_find_text_file, "w") as text_file:
                for batch_num in range(num_images // batch_size):
                    batch = next(loader).cuda(cuda_rank)

                    encoder_output = stylex.encoder(batch)

                    # noise = image_noise(batch_size, image_size, device=cuda_rank)

                    real_classified_logits = classifier.classify_images(batch)

                    latent_w = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                                 stylex.G.num_layers)]  # Has to be bracketed because expects a noise mix

                    # dlatents.append(encoder_output)
                    w_latent_tensor = styles_def_to_tensor(latent_w)

                    generated_images, style_coords = stylex.G(w_latent_tensor, zero_noise, get_style_coords=True)

                    base_prob_logits = classifier.classify_images(generated_images)

                    feature = {}

                    style_change_effect = torch.zeros(batch_size, style_vector_amount, 2, 2).cuda(cuda_rank)
                    individual_shift_tensor = torch.zeros(batch_size, style_vector_amount, 2).cuda(cuda_rank)

                    for sindex in tqdm.tqdm(range(style_vector_amount)):
                        block_idx, weight_idx = sindex_to_block_idx_and_index(stylex.G, sindex)

                        block = stylex.G.blocks[block_idx]

                        # block.style1.bias [block.input_channels]
                        # block.style2.bias [block.filters]

                        current_style_layer = None
                        one_hot = None

                        if weight_idx < block.input_channels:
                            # While we're in style 1.
                            current_style_layer = block.to_style1
                            one_hot = torch.zeros((batch_size, block.input_channels)).cuda(cuda_rank)
                        else:
                            weight_idx -= block.input_channels
                            current_style_layer = block.to_style2
                            one_hot = torch.zeros((batch_size, block.filters)).cuda(cuda_rank)

                        one_hot[:, weight_idx] = 1

                        s_shift_down = one_hot * (
                                (minimums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(1)
                        s_shift_up = one_hot * ((maximums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(
                            1)

                        for direction_index, shift in enumerate([s_shift_down, s_shift_up]):
                            for image_index, individual_shift in enumerate(shift):
                                current_style_layer.bias += individual_shift
                                perturbed_generated_images, style_coords = stylex.G(
                                    w_latent_tensor[image_index].unsqueeze(0),
                                    zero_noise[image_index].unsqueeze(0),
                                    get_style_coords=True)
                                shift_classification = classifier.classify_images(perturbed_generated_images)
                                style_change_effect[image_index, sindex, direction_index] = shift_classification - \
                                                                                            base_prob_logits[
                                                                                                image_index]
                                if direction_index == 0:
                                    shift_save = (minimums[sindex] - style_coords[:, sindex]) * s_shift_size
                                else:
                                    shift_save = (maximums[sindex] - style_coords[:, sindex]) * s_shift_size

                                individual_shift_tensor[image_index, sindex, direction_index] = shift_save                                                                
                                current_style_layer.bias -= individual_shift

                    if batch_num == 0:
                        style_changes_to8 = style_change_effect[0, :, :, 1] - style_change_effect[0, :, :, 0]

                        # style_changes_norm = torch.linalg.norm(style_changes_test, dim=2, keepdim=True)

                        # Destroy the direction dimension
                        style_changes_norm_flattened = style_changes_to8.flatten()

                        # Sort the style_changes by the norm
                        style_changes_norm_sorted, style_changes_indices = torch.sort(style_changes_norm_flattened,
                                                                                      descending=True)
                        print(style_changes_norm_sorted[0], style_changes_indices[0])
                        
                        print("top shift tensor", "direction:", style_changes_indices[0] % 2, "tensor:", individual_shift_tensor[0, style_changes_indices[0] / 2, style_changes_indices[0] % 2])

                    feature['base_prob_logits'] = base_prob_logits.tolist()  # .flatten()
                    feature['wlatent'] = w_latent_tensor[:, 0].tolist()  # .flatten()
                    feature['style_change_effect'] = style_change_effect.tolist()  # .flatten())
                    print(feature, file=text_file)
                    # text_file.write("\n")

        # style_change_effect[batch_size, 1, sindex, :]

        total_number_of_batches = num_images // batch_size
        style_change_effect = torch.zeros(total_number_of_batches, batch_size, style_vector_amount, 2, 2).cuda(cuda_rank)
        wlatents = torch.zeros(total_number_of_batches, batch_size, stylex.G.latent_dim).cuda(cuda_rank)
        base_probs = torch.zeros(total_number_of_batches, batch_size, 2).cuda(cuda_rank)

        # Load the file and get the data
        with open(att_find_text_file, "r") as text_file:
            # Read a line and import it as a dictionary
            for batch_number, line in enumerate(text_file):
                # feature_dict = dict(line.readline())
                feature_dict = ast.literal_eval(line)

                # Get the wlatent
                wlatent = feature_dict['wlatent']
                wlatents[batch_number] = torch.tensor(wlatent)

                # Get the base_prob_logits
                base_prob_logits = feature_dict['base_prob_logits']
                base_probs[batch_number] = torch.tensor(base_prob_logits)

                # Get the result
                result = feature_dict['style_change_effect']
                style_change_effect[batch_number] = torch.tensor(result)

        # Reshape the first two dimensions to the amount of images
        num_images = total_number_of_batches * batch_size
        style_change_effect = style_change_effect.view(num_images, style_vector_amount, 2, 2)

        wlatents = wlatents.view(num_images, stylex.G.latent_dim)
        base_probs = base_probs.view(num_images, 2)

        # Tadija's idea: Plot the 4 images with the highest change in classifications

        # style_change_effect[total_num_images, style_vector_amount, 2, 2)
        # noise = image_noise(1, image_size, device=cuda_rank)
        zero_noise = torch.zeros(1, image_size, image_size, 1).cuda(cuda_rank)
        wlatents = wlatents.unsqueeze(1).expand(4, stylex.G.num_layers, stylex.G.latent_dim)

        wlatent_test = wlatents[0]

        img, style_coords = stylex.G(wlatent_test.unsqueeze(0), zero_noise, get_style_coords=True)

        import matplotlib.pyplot as plt
        plt.title("Original image")
        plt.imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0))
        plt.show()

        style_changes_test = style_change_effect[0]

        # Calculate the difference between the 2 digits in the last layer

        # [1184 2 2]

        # [1184 2 1]

        style_changes_to8 = style_changes_test[:, :, 1] - style_changes_test[:, :, 0]

        # style_changes_norm = torch.linalg.norm(style_changes_test, dim=2, keepdim=True)

        # Destroy the direction dimension
        style_changes_norm_flattened = style_changes_to8.flatten()

        # Sort the style_changes by the norm
        style_changes_norm_sorted, style_changes_indices = torch.sort(style_changes_norm_flattened, descending=True)

        # [style_coords, direction, num_class]
        # [style_coords, direction]
        # sort decreasing on num_class

        k = 1
        one_hot = None

        top_k_changes, top_k_sindex, top_k_directions = style_changes_norm_sorted[:k], style_changes_indices[
                                                                                       :k] / 2, style_changes_indices[
                                                                                                :k] % 2

        # Change the biases in the direction specified.
        for i in range(k):

            sindex = int(top_k_sindex[i])

            block_idx, weight_idx = sindex_to_block_idx_and_index(stylex.G, sindex)

            block = stylex.G.blocks[block_idx]
            if weight_idx < block.input_channels:
                # While we're in style 1.
                current_style_layer = block.to_style1
                one_hot = torch.zeros((1, block.input_channels)).cuda(cuda_rank)
            else:
                weight_idx -= block.input_channels
                current_style_layer = block.to_style2
                one_hot = torch.zeros((1, block.filters)).cuda(cuda_rank)

            one_hot[:, weight_idx] = 1

            if top_k_directions[i] == 0:
                individual_shift = one_hot * (
                        (minimums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(1)
                save_shift = (minimums[sindex] - style_coords[:, sindex]) * s_shift_size
            else:
                individual_shift = one_hot * (
                        (maximums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(1)
                save_shift = (maximums[sindex] - style_coords[:, sindex]) * s_shift_size

            current_style_layer.bias += individual_shift[0]
            print("shift:", save_shift[0], "direction:", top_k_directions[i])

        # Generate the perturbed image
        perturbed_generated_images, style_coords = stylex.G(wlatent_test.unsqueeze(0),
                                                            zero_noise,
                                                            get_style_coords=True)
        shift_classification = classifier.classify_images(perturbed_generated_images)
        print("Base classification encoded image:", base_probs[0])
        print("New classification:", shift_classification)
        print("Difference:", shift_classification - base_probs[0])
        print("Expected difference:", style_changes_test[int(top_k_sindex[0]), int(top_k_directions[0])])
        print("Biggest style change and index:", style_changes_norm_sorted[0], style_changes_indices[0])

        # Plot the perturbed images
        plot_image(perturbed_generated_images[0])

        # Change the biases back to their original direction
        for i in range(k):
            current_style_layer.bias -= individual_shift

        perturbed_image = stylex.G(wlatent_test, zero_noise)

        plt.title("Changed 10 attributes image")
        plt.imshow(perturbed_image)
        plt.show()

        # top-10 sindexes for image 0 

        # G(original)
        # change the biases of the top-10 attributes
        # G(changed)

        saved_change_effect = filter_unstable_images(style_change_effect)

        generated_images, style_coords = stylex.G(wlatents, zero_noise, get_style_coords=True)

        """
        all_style_vectors = tf.concat(generator.style_vector_calculator(W_values, training=False)[0], axis=1).numpy()
        style_min = np.min(all_style_vectors, axis=0)
        style_max = np.max(all_style_vectors, axis=0)

        all_style_vectors_distances = np.zeros((all_style_vectors.shape[0], all_style_vectors.shape[1], 2))
        all_style_vectors_distances[:,:, 0] = all_style_vectors - np.tile(style_min, (all_style_vectors.shape[0], 1))
        all_style_vectors_distances[:,:, 1] = np.tile(style_max, (all_style_vectors.shape[0], 1)) - all_style_vectors

        # The guy she tells you not to worry about: [batch_size, 2, num_style_coords, num_classes]
        # You: [batch_size, 2, 1184, 2]

        #   pass style vector to generator
        #   classify the result and save it to base_prob
        #   for sindex in range(minimums.shape):

        # style_change_effect: A shape of [num_images, 2, style_size, num_classes].
        #   The effect of each change of style on specific direction on each image.

        # Original tensorflow code
        

        #print(style_change_effect.shape) # num_images, direction, sindex, logits
        #print(dlatents.shape) # num_images, latent_index
        #print(base_probs.shape) # num_images, base_logits

        #@title Split by class
        all_labels = np.argmax(base_probs, axis=1)
        style_effect_classes = {}
        W_classes = {}
        style_vectors_distances_classes = {}
        all_style_vectors_classes = {}
        for img_ind in range(label_size):
            img_inx = np.array([i for i in range(all_labels.shape[0]) 
            if all_labels[i] == img_ind])
            curr_style_effect = np.zeros((len(img_inx), style_change_effect.shape[1], style_change_effect.shape[2], style_change_effect.shape[3]))
            curr_w = np.zeros((len(img_inx), W_values.shape[1]))
            curr_style_vector_distances = np.zeros((len(img_inx), style_change_effect.shape[2], 2))
            for k, i in enumerate(img_inx):
                curr_style_effect[k, :, :] = style_change_effect[i, :, :, :]
                curr_w[k, :] = W_values[i, :]
                curr_style_vector_distances[k, :, :] = all_style_vectors_distances[i, :, :]
            style_effect_classes[img_ind] = curr_style_effect
            W_classes[img_ind] = curr_w
            style_vectors_distances_classes[img_ind] = curr_style_vector_distances
            all_style_vectors_classes[img_ind] = all_style_vectors[img_inx]
            print(f'Class {img_ind}, {len(img_inx)} images.')
            #@title Load effect data from the tfrecord {form-width: '20%'}
            
        """

        #   pass style vector to generator
        #   classify the result and save it to base_prob
        #   for sindex in range(minimums.shape):
        #       !!! We need some function to get the layer number and index number from the index !!!
        #

        # Get the latent vectors of the images w.

        # dlatents = []
        # w_styles = []
