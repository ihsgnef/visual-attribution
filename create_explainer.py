from explainer import backprop as bp
from explainer import deeplift as df
from explainer import sparse_deeplift as sdf
from explainer import gradcam as gc
from explainer import patterns as pt
from explainer import ebp
from explainer import real_time as rt
from explainer import sparse as sparse
from explainer import vat


def get_explainer(model, name, extra_args):
    methods = {
        'vanilla_grad': bp.VanillaGradExplainer,
        'vanilla_grad_new': bp.VanillaGradNewExplainer,
        'grad_x_input': bp.GradxInputExplainer,
        'grad_x_input_new': bp.GradxInputNewExplainer,
        'saliency': bp.SaliencyExplainer,
        'integrate_grad': bp.IntegrateGradExplainer,
        'deconv': bp.DeconvExplainer,
        'guided_backprop': bp.GuidedBackpropExplainer,
        'deeplift_rescale': df.DeepLIFTRescaleExplainer,
        'deeplift_rescale_sparse': sdf.SparseDeepLIFTRescaleExplainer,
        'gradcam': gc.GradCAMExplainer,
        'pattern_net': pt.PatternNetExplainer,
        'pattern_lrp': pt.PatternLRPExplainer,
        'excitation_backprop': ebp.ExcitationBackpropExplainer,
        'contrastive_excitation_backprop': ebp.ContrastiveExcitationBackpropExplainer,
        'real_time_saliency': rt.RealTimeSaliencyExplainer,
        'sparse': sparse.SparseExplainer,
        'sparse_integrate_grad': bp.SparseIntegrateGradExplainer,
        'sparse_guided_backprop': bp.SparseGuidedBackpropExplainer,
        'vat': vat.VATExplainer,
    }

    name = name.split(' ')[0]

    if name == 'smooth_grad':
        base_explainer = methods['vanilla_grad'](model)
        explainer = bp.SmoothGradExplainer(base_explainer)

    elif name == 'sparse_smooth_grad':
        base_explainer = methods['sparse'](model)
        explainer = bp.SmoothGradExplainer(base_explainer)

    elif name.find('pattern') != -1:
        explainer = methods[name](
            model,
            params_file='./weights/imagenet_224_vgg_16.npz',
            pattern_file='./weights/imagenet_224_vgg_16.patterns.A_only.npz'
        )

    elif name == 'gradcam':
        if model.__class__.__name__ == 'VGG':
            explainer = methods[name](
                model, target_layer_name_keys=['features', '30']  # pool5
            )
        elif model.__class__.__name__ == 'ResNet':
            explainer = methods[name](
                model, target_layer_name_keys=['avgpool'], use_inp=True,
            )

    elif name == 'excitation_backprop':
        if model.__class__.__name__ == 'VGG':  # vgg16
            explainer = methods[name](
                model,
                output_layer_keys=['features', '23']  # pool4
            )
        elif model.__class__.__name__ == 'ResNet':  # resnet50
            explainer = methods[name](
                model,
                output_layer_keys=['layer4', '1', 'conv1']  # res4a
            )

    elif name == 'contrastive_excitation_backprop':
        if model.__class__.__name__ == 'VGG':  # vgg16
            explainer = methods[name](
                model,
                intermediate_layer_keys=['features', '30'],  # pool5
                output_layer_keys=['features', '23'],  # pool4
                final_linear_keys=['classifier', '6']  # fc8
            )
        elif model.__class__.__name__ == 'ResNet':  # resnet50
            explainer = methods[name](
                model,
                intermediate_layer_keys=['avgpool'],
                output_layer_keys=['layer4', '1', 'conv1'],  # res4a
                final_linear_keys=['fc']
            )
    elif name == 'real_time_saliency':
        explainer = methods[name]('./weights/model-1.ckpt')

    else:
        if extra_args is not None:
            explainer = methods[name](model, **extra_args)
        else:
            explainer = methods[name](model)

    return explainer


def get_heatmap(saliency):
    saliency = saliency.squeeze()

    if len(saliency.size()) == 2:
        return saliency.abs().cpu().numpy()
    else:
        return saliency.abs().max(0)[0].cpu().numpy()
