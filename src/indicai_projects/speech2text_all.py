import torch
import numpy as np
import json
import onnxruntime
import librosa
from pathlib import Path

class Indic_STT_ALL:
    def __init__(self,model_path,device):
        self.BLANK_ID = 256
        self.RNNT_MAX_SYMBOLS = 10
        self.PRED_RNN_LAYERS = 2
        self.PRED_RNN_HIDDEN_DIM = 640
        self.SOS = 256
        self.supported_langs = ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if "cuda" in str(device) else ['CPUExecutionProvider']
        self.joint_enc = onnxruntime.InferenceSession(Path(model_path,"joint_enc.onnx"), providers = provider)
        self.joint_pred = onnxruntime.InferenceSession(Path(model_path,"joint_pred.onnx"), providers = provider)
        self.joint_pre_net = onnxruntime.InferenceSession(Path(model_path,"joint_pre_net.onnx"), providers = provider)
        self.rnnt_decoder = onnxruntime.InferenceSession(Path(model_path,"rnnt_decoder.onnx"), providers = provider)
        self.ctc_decoder = onnxruntime.InferenceSession(Path(model_path,"ctc_decoder.onnx"), providers = provider)
        self.models = {n:onnxruntime.InferenceSession(Path(model_path,f'joint_post_net_{n}.onnx'), providers = provider) for n in self.supported_langs}
        self.encoder = onnxruntime.InferenceSession(Path(model_path,"encoder_model/encoder.onnx"), providers = provider)
        self.device = device
        self.preprocessor = torch.jit.load(Path(model_path,'preprocessor.ts'), map_location=self.device)
        self.vocab = json.load(open(Path(model_path,'vocab.json')))
        self.language_masks = json.load(open(Path(model_path,'language_masks.json')))
    def ctc_decode(self, encoder_outputs, lang):
        logprobs = self.ctc_decoder.run(['logprobs'], {'encoder_output': encoder_outputs})[0]
        logprobs = torch.from_numpy(logprobs[:, :, self.language_masks[lang]]).log_softmax(dim=-1)
        indices = torch.argmax(logprobs[0],dim=-1)
        collapsed_indices = torch.unique_consecutive(indices, dim=-1)
        hyp = ''.join([self.vocab[lang][x] for x in collapsed_indices if x != self.BLANK_ID]).replace('▁',' ').strip()
        del logprobs, indices, collapsed_indices
        return hyp

    def rnnt_decode(self, encoder_outputs, lang):
        joint_enc = self.joint_enc.run(['output'], {'input': encoder_outputs.transpose(0, 2, 1)})[0]
        joint_enc = torch.from_numpy(joint_enc)
        hyp = [self.SOS]
        prev_dec_state = (np.zeros((self.PRED_RNN_LAYERS, 1, self.PRED_RNN_HIDDEN_DIM), dtype=np.float32),np.zeros((self.PRED_RNN_LAYERS, 1, self.PRED_RNN_HIDDEN_DIM), dtype=np.float32))
        
        # Iterate over time steps (T)
        for t in range(joint_enc.size(1)):
            f = joint_enc[:, t, :].unsqueeze(1)  # B x 1 x H

            not_blank = True
            symbols_added = 0

            while not_blank and ((self.RNNT_MAX_SYMBOLS is None) or (symbols_added < self.RNNT_MAX_SYMBOLS)):
                # Decoder forward passsaa
                g, _, dec_state_0, dec_state_1 = self.rnnt_decoder.run(
                    ['outputs', 'prednet_lengths', 'states', '162'],
                    {'targets': np.array([[hyp[-1]]], dtype=np.int32),
                     'target_length': np.array([1], dtype=np.int32),
                     'states.1': prev_dec_state[0],
                     'onnx::Slice_3': prev_dec_state[1]})
                
                # Joint network
                g = self.joint_pred.run(['output'], {'input': g.transpose(0,2,1)})[0]

                joint_out = f + g  # B x 1 x H
                joint_out = self.joint_pre_net.run(['output'], {'input': joint_out.numpy()})[0]
            
                logits = self.models[lang].run(['output'], {'input': joint_out})[0]
                log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                pred_token = log_probs.argmax(dim=-1).item()

                # Append if not blank
                if pred_token == self.BLANK_ID:
                    not_blank = False
                else:
                    hyp.append(pred_token)
                    prev_dec_state = (dec_state_0, dec_state_1)

                symbols_added += 1

        pred_text = ''.join([self.vocab[lang][x] for x in hyp if x != self.SOS]).replace('▁',' ').strip()
        return pred_text

    def predict(self, audio_path, lang):
        wav = torch.tensor(librosa.load(audio_path, sr=16000, mono=True)[0])
        audio_signal, length = self.preprocessor(input_signal=wav.unsqueeze(0).to(self.device), length=torch.tensor([wav.shape[-1]]).to(self.device))
        encoder_outputs = self.encoder.run(['outputs', 'encoded_lengths'], {'audio_signal': audio_signal.cpu().numpy(), 'length': length.cpu().numpy()})[0]
        return self.ctc_decode(encoder_outputs, lang),self.rnnt_decode(encoder_outputs, lang)
    