import numpy as np
import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P

def main():
    model_methods = [
        #['resnet50', 'vanilla_grad', 'imshow', None],
        ['resnet50', 'grad_x_input', 'imshow', None],
        #['resnet50', 'sparse_guided_backprop', 'imshow', None],
        ['resnet18', 'sparse', 'imshow', None], ########## resnet50        
        # ['resnet50', 'saliency', 'imshow', None],
        # ['resnet50', 'sparse_integrate_grad', 'imshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        #['resnet50', 'guided_backprop', 'imshow', None],
        ['resnet50', 'smooth_grad', 'imshow', None],
        ['resnet50', 'sparse_smooth_grad', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
        # ['vgg16', 'pattern_net', 'imshow', None],
        # ['vgg16', 'pattern_lrp', 'camshow', None],
        # ['resnet50', 'real_time_saliency', 'camshow', None],
        ]

    image_path = 'examples/fox.png'
    raw_img = viz.pil_loader(image_path)
    all_saliency_maps = []
    for model_name, method_name, _, kwargs in model_methods:
        print(method_name)
        transf = get_preprocess(model_name, method_name)
        model = utils.load_model(model_name)        
        model.cuda()
        explainer = get_explainer(model, method_name, kwargs)
        inp = transf(raw_img)
        inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

        saliency = explainer.explain(inp, None)
        saliency = viz.VisualizeImageGrayscale(saliency)            
        all_saliency_maps.append(saliency.cpu().numpy()[0])

    plt.figure(figsize=(25, 15))
    plt.subplot(3, 5, 1)
    plt.imshow(raw_img)
    plt.axis('off')
    plt.title('Tricycle')
    for i, saliency in enumerate(all_saliency_maps):
        model_name, method_name, show_style, extra_args = model_methods[i]
        plt.subplot(3, 5, i + 2 + i // 4)
        # if show_style == 'camshow':
        #     viz.plot_cam(utils.upsample(np.expand_dims(saliency, axis=0),
        #                                 (raw_img.height, raw_img.width)),
        #                  raw_img, 'jet', alpha=0.5)        
        # else:
        plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)

        plt.axis('off')        
        plt.title(method_name)
    plt.tight_layout()
    plt.savefig('output/tusker_saliency.png')

if __name__ == '__main__':
    main()    