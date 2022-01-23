# For the attribute find functions.
from stylex_train import StylEx, Dataset, DistributedSampler, MNIST_1vA

# For setting data src.
from stylex_train import image_noise, styles_def_to_tensor, make_weights_for_balanced_classes, cycle, default

from mobilenet_classifier import MobileNet
import torch
from torch.utils import data
import math
import tqdm

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


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

        noise = image_noise(batch_size, image_size, device=cuda_rank)

        latent_w = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                     stylex.G.num_layers)]  # Has to be bracketed because expects a noise mix

        # dlatents.append(encoder_output)
        w_latent_tensor = styles_def_to_tensor(latent_w)

        rgb, style_coords = stylex.G(w_latent_tensor, noise, get_style_coords=True)

        if minimums is None or maximums is None:
            minimums = style_coords
            maximums = style_coords
        else:
            minimums = torch.minimum(minimums, style_coords)
            maximums = torch.maximum(maximums, style_coords)

    minimums = torch.min(minimums, dim=0)[0]
    maximums = torch.max(maximums, dim=0)[0]

    return minimums, maximums


def run_attrfind(
        data='./',
        stylex_path='',
        classifier_name='',
        image_size=32,
        num_images=4,
        batch_size=16,
        dataset_name=None,
        s_shift_size=1
):
    with torch.no_grad():
        dataset, loader = set_data_src(data, dataset_name, image_size, batch_size)
        # Since attribute find is not helped by having multiple GPU's, we hardcode cuda_rank to 0.
        cuda_rank = 0

        stylex = StylEx(image_size=image_size)
        classifier = MobileNet(classifier_name, cuda_rank=cuda_rank, output_size=2, image_size=image_size)

        minimums, maximums = get_min_max_style_vectors(stylex, classifier, loader, batch_size=batch_size,
                                                       cuda_rank=cuda_rank, image_size=image_size)

        # Dictionary of features
        feature = {}  # For later TODO: Implement

        style_change_effect = torch.zeros(batch_size, 2, len(minimums), 2)

        for _ in range(num_images // batch_size):
            batch = next(loader).cuda(cuda_rank)

            encoder_output = stylex.encoder(batch)

            noise = image_noise(batch_size, image_size, device=cuda_rank)

            real_classified_logits = classifier.classify_images(batch)

            latent_w = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                         stylex.G.num_layers)]  # Has to be bracketed because expects a noise mix

            # dlatents.append(encoder_output)
            w_latent_tensor = styles_def_to_tensor(latent_w)

            generated_images, style_coords = stylex.G(w_latent_tensor, noise, get_style_coords=True)

            base_prob_logits = classifier.classify_images(generated_images)

            classifier_results = []

            style_vector_amount = len(minimums)

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

                s_shift_down = one_hot * ((minimums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(1)
                s_shift_up = one_hot * ((maximums[sindex] - style_coords[:, sindex]) * s_shift_size).unsqueeze(1)

                for direction_index, shift in enumerate([s_shift_down, s_shift_up]):
                    for image_index, individual_shift in enumerate(shift):
                        current_style_layer.bias += individual_shift
                        perturbed_generated_images, style_coords = stylex.G(w_latent_tensor[image_index].unsqueeze(0),
                                                                            noise[image_index].unsqueeze(0),
                                                                            get_style_coords=True)
                        shift_classification = classifier.classify_images(perturbed_generated_images)
                        classifier_results.extend(shift_classification)  # TODO: reshape
                        current_style_layer.bias -= individual_shift

                        style_change_effect[image_index, direction_index, sindex] = shift_classification - base_prob_logits[image_index]

        # style_change_effect[batch_size, 1, sindex, :]

        print(style_change_effect)

        # The guy she tells you not to worry about: [batch_size, 2, num_style_coords, num_classes]
        # You: [batch_size, 2, 1184, 2]

        #   pass style vector to generator
        #   classify the result and save it to base_prob
        #   for sindex in range(minimums.shape):

        # style_change_effect: A shape of [num_images, 2, style_size, num_classes].
        #   The effect of each change of style on specific direction on each image.

        # Original tensorflow code
        """
        with tf.io.TFRecordWriter(data_path) as writer:
        for dlatent_index, dlatent in dlatents:
            print("Image ", dlatent_index)
            expanded_dlatent = tf.tile(tf.expand_dims(dlatent, 1), [1, num_layers, 1])
            base_prob = get_classifier_results(generator, expanded_dlatent)
            classifier_results = []
            print("going to loop over number of indices = ", s_indices_num)
            for sindex in tqdm.tqdm(range(0, s_indices_num)):
                layer_idx, weight_idx = sindex_to_layer_idx_and_index(generator, sindex)
                print(layer_idx, weight_idx)
                layer = generator.style_vector_calculator.style_dense_blocks[layer_idx]
                layer_size = layer.dense_bias.weights[0].shape[1]
                # Get the style vector.
                s_vals = tf.concat(generator.style_vector_calculator(dlatent, training=False)[0], axis=1).numpy()[0]
                print("Original style vector", s_vals.shape)
                s_shift_down = (minimums[sindex] - s_vals[sindex]) * s_shift_size
                s_shift_up = (maximums[sindex] - s_vals[sindex]) * s_shift_size
                
                s_shift_d = s_shift_down * tf.expand_dims(tf.one_hot(weight_idx, 
                                                              layer_size), axis=0)
                print("Change value in layer", s_shift_d)
                layer.dense_bias.weights[0].assign_add(s_shift_d)
                classifier_results.extend(get_classifier_results(generator, expanded_dlatent) - base_prob)
                layer.dense_bias.weights[0].assign_add(-s_shift_d)
                
                s_shift_u = s_shift_up * tf.expand_dims(tf.one_hot(weight_idx, layer_size), axis=0)
    
                
                print("Change value in layer", s_shift_u)
                layer.dense_bias.weights[0].assign_add(s_shift_u)
                classifier_results.extend(get_classifier_results(generator, expanded_dlatent) - base_prob)
                layer.dense_bias.weights[0].assign_add(-s_shift_u)
                break
            break
    
            feature = {}
            feature['base_prob'] = _float_features(base_prob.flatten())
            feature['dlatent'] = _float_features(dlatent.flatten())
            feature['result'] = _float_features(np.array(classifier_results).flatten())
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())
        """

        #   pass style vector to generator
        #   classify the result and save it to base_prob
        #   for sindex in range(minimums.shape):
        #       !!! We need some function to get the layer number and index number from the index !!!
        #

        # Get the latent vectors of the images w.

        # dlatents = []
        # w_styles = []
