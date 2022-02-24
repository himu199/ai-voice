dependencies = ['torch']
import torch
from model.generator import Generator

model_params = {
    'nvidia_tacotron2_LJ11_epoch6400': {
        'mel_channel': 80,
        'model_path': ' /weights/nvidia_tacotron2_LJ11_epoch6400.ptweights',
    },
}


def melgan(model_name='nvidia_tacotron2_LJ11_epoch6400', pretrained=True, progress=True):
    params = model_params[model_name]
    model = Generator(params['mel_channel'])

    if pretrained:
        state_dict = torch.hub.load(params['model_path'],
                                    source='local')
        model.load_state_dict(state_dict['model_g'])

    model.eval(inference=True)

    return model


if __name__ == '__main__':
    vocoder = torch.hub.load('/pdf2audio/melgan-master', 'melgan')
    mel = torch.randn(1, 80, 234) # use your own mel-spectrogram here

    print('Input mel-spectrogram shape: {}'.format(mel.shape))

    if torch.cuda.is_available():
        print('Moving data & model to GPU')
        vocoder = vocoder.cuda()
        mel = mel.cuda()

    with torch.no_grad():
        audio = vocoder.inference(mel)

    print('Output audio shape: {}'.format(audio.shape))
