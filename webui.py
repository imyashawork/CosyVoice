# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import tempfile
import subprocess
import time
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

SYSTEM_PROMPT = 'You are a helpful assistant.'
END_PROMPT = '<|endofprompt|>'

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制', '语音转换']

instruct_dict = {'预训练音色': '1. 选择预训练音色（如有）\n2. 输入合成文本\n3. 点击生成音频按钮\n注意: CosyVoice3 无预训练音色，请使用其他模式',
                 '3s极速复刻': '1. 上传或录制 prompt 音频（不超过 30 秒，采样率不低于 16kHz）\n'
                              '2. 输入 prompt 文本（需与音频内容一致）\n'
                              '   CosyVoice3 格式: You are a helpful assistant.<|endofprompt|>音频对应文字\n'
                              '3. 输入合成文本\n'
                              '4. 点击生成音频按钮\n'
                              '提示: 可在合成文本中插入 [j][ǐ] 等拼音标记进行多音字纠错',
                 '跨语种复刻': '1. 上传或录制 prompt 音频（不超过 30 秒）\n'
                              '2. 输入合成文本（需包含 CosyVoice3 格式前缀）\n'
                              '   CosyVoice3 格式: You are a helpful assistant.<|endofprompt|>合成内容\n'
                              '   支持细粒度控制: [breath] [laughter] <strong>...</strong> 等\n'
                              '   日语需转换为片假名，如: レキシ テキ セカイ\n'
                              '3. 点击生成音频按钮',
                 '自然语言控制': '1. 上传或录制 prompt 音频（目标音色）\n'
                              '2. 从指令预设下拉框选择，或手动输入指令\n'
                              '   CosyVoice3 格式: You are a helpful assistant. <指令><|endofprompt|>\n'
                              '   支持: 方言切换、语速控制、情感表达、风格模仿等\n'
                              '3. 输入合成文本\n'
                              '4. 点击生成音频按钮',
                 '语音转换': '1. 上传源音频（提供语音内容）\n'
                            '2. 上传或录制 prompt 音频（提供目标音色）\n'
                            '3. 点击生成音频按钮（不需要输入文本）'}

instruct_preset_choices = [
    '请用广东话表达。',
    '请用四川话表达。',
    '请用东北话表达。',
    '请用上海话表达。',
    '请用天津话表达。',
    '请用陕西话表达。',
    '请用山东话表达。',
    '请用河南话表达。',
    '请用湖南话表达。',
    '请用闽南话表达。',
    '请用云南话表达。',
    '请用山西话表达。',
    '请用甘肃话表达。',
    '请用贵州话表达。',
    '请用湖北话表达。',
    '请用江西话表达。',
    '请用宁夏话表达。',
    '请用尽可能快地语速说一句话。',
    '请用尽可能慢地语速说一句话。',
    'Please say a sentence as loudly as possible.',
    'Please say a sentence in a very soft voice.',
    '请非常开心地说一句话。',
    '请非常伤心地说一句话。',
    '请非常生气地说一句话。',
    '我想体验一下小猪佩奇风格，可以吗？',
    '你可以尝试用机器人的方式解答吗？',
]

stream_mode_list = [('否', False), ('是', True)]
device_mode_list = [('GPU (CUDA)', 'cuda'), ('CPU', 'cpu')]
max_val = 0.8


def reload_model(device):
    global cosyvoice, sft_spk, default_data
    device_label = 'GPU (CUDA)' if device == 'cuda' else 'CPU'
    gr.Info('正在切换到{}，模型重新加载中，请等待...'.format(device_label))
    del cosyvoice
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cosyvoice = AutoModel(model_dir=args.model_dir, device=device, fp16=args.fp16)
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    default_data = np.zeros(cosyvoice.sample_rate)
    gr.Info('已成功切换到{}'.format(device_label))
    return gr.update(choices=sft_spk, value=sft_spk[0] if sft_spk and sft_spk[0] else None)


def stop_audio_generation():
    cosyvoice.model.stop_signal = True
    gr.Info('正在停止音频生成...')


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def select_instruct_preset(preset_choice):
    if preset_choice == '' or preset_choice is None:
        return {"__type__": "update", "value": ''}
    full_instruct = '{} {}{}'.format(SYSTEM_PROMPT, preset_choice, END_PROMPT)
    return {"__type__": "update", "value": full_instruct}


def ensure_endofprompt(text, is_instruct=False):
    if text is None or text == '':
        return text
    if END_PROMPT in text:
        return text
    if is_instruct:
        return '{} {}{}'.format(SYSTEM_PROMPT, text, END_PROMPT)
    else:
        return '{}{}{}'.format(SYSTEM_PROMPT, END_PROMPT, text)


def save_output_audio(chunks, sample_rate):
    if not chunks:
        return None
    final_audio = np.concatenate(chunks)
    output_dir = os.path.join(ROOT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, 'tts_{}.wav'.format(timestamp))
    torchaudio.save(output_path, torch.from_numpy(final_audio).unsqueeze(0).float(), sample_rate)
    logging.info('audio saved to {}'.format(output_path))
    return output_path


def trim_prompt_wav(wav_path, max_duration=30.0):
    if wav_path is None:
        return wav_path
    try:
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate
    except Exception:
        return wav_path

    if duration <= max_duration:
        return wav_path

    # ffmpeg: 去除开头静音后截取 max_duration 秒
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=tempfile.gettempdir())
        tmp.close()
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-af', 'silenceremove=start_periods=1:start_silence=0.1:start_threshold=-30dB',
            '-t', str(max_duration),
            '-acodec', 'pcm_s16le',
            tmp.name
        ], capture_output=True, check=True, timeout=60)
        logging.info('prompt audio trimmed from {:.1f}s to {}s'.format(duration, max_duration))
        return tmp.name
    except Exception:
        pass

    # 回退: librosa 去除静音 + 截取
    try:
        sr = info.sample_rate
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        max_samples = int(max_duration * sr)
        if len(y_trimmed) > max_samples:
            y_trimmed = y_trimmed[:max_samples]
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=tempfile.gettempdir())
        tmp.close()
        torchaudio.save(tmp.name, torch.from_numpy(y_trimmed).unsqueeze(0).float(), sr)
        logging.info('prompt audio trimmed from {:.1f}s to {}s (librosa)'.format(duration, max_duration))
        return tmp.name
    except Exception as e:
        logging.warning('failed to trim prompt audio: {}'.format(e))
        return wav_path


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text,
                   prompt_wav_upload, prompt_wav_record, source_wav_upload,
                   instruct_text, seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    source_wav = source_wav_upload
    model_class = cosyvoice.__class__.__name__

    # 自动截断超过 30 秒的音频
    prompt_wav = trim_prompt_wav(prompt_wav)
    source_wav = trim_prompt_wav(source_wav)

    # CosyVoice3: 自动补全 <|endofprompt|> 格式
    if model_class == 'CosyVoice3':
        if mode_checkbox_group == '3s极速复刻':
            prompt_text = ensure_endofprompt(prompt_text)
        elif mode_checkbox_group == '跨语种复刻':
            tts_text = ensure_endofprompt(tts_text)
        elif mode_checkbox_group == '自然语言控制':
            instruct_text = ensure_endofprompt(instruct_text, is_instruct=True)

    # --- 预训练音色 (SFT) ---
    if mode_checkbox_group == '预训练音色':
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '' or sft_dropdown is None:
            gr.Warning('没有可用的预训练音色！此模型可能不支持SFT模式，请尝试其他模式。')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        chunks = [i['tts_speech'].numpy().flatten()
                  for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)]
        output_path = save_output_audio(chunks, cosyvoice.sample_rate)
        if output_path:
            gr.Info('音频已保存至: {}'.format(output_path))
            yield output_path, output_path
        else:
            yield (cosyvoice.sample_rate, default_data), ''

    # --- 3s极速复刻 (Zero-shot) ---
    elif mode_checkbox_group == '3s极速复刻':
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        try:
            if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
                gr.Warning('prompt音频采样率{}低于{}'.format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr))
                yield (cosyvoice.sample_rate, default_data), ''
                return
        except Exception:
            gr.Warning('prompt音频文件无法读取，请检查文件格式')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，instruct文本会被忽略！')
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        chunks = [i['tts_speech'].numpy().flatten()
                  for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav,
                                                          stream=stream, speed=speed)]
        output_path = save_output_audio(chunks, cosyvoice.sample_rate)
        if output_path:
            gr.Info('音频已保存至: {}'.format(output_path))
            yield output_path, output_path
        else:
            yield (cosyvoice.sample_rate, default_data), ''

    # --- 跨语种复刻 (Cross-lingual) ---
    elif mode_checkbox_group == '跨语种复刻':
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式，instruct文本会被忽略！')
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        chunks = [i['tts_speech'].numpy().flatten()
                  for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav,
                                                              stream=stream, speed=speed)]
        output_path = save_output_audio(chunks, cosyvoice.sample_rate)
        if output_path:
            gr.Info('音频已保存至: {}'.format(output_path))
            yield output_path, output_path
        else:
            yield (cosyvoice.sample_rate, default_data), ''

    # --- 自然语言控制 (Instruct) ---
    elif mode_checkbox_group == '自然语言控制':
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式，请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        if model_class in ('CosyVoice2', 'CosyVoice3'):
            if prompt_wav is None:
                gr.Warning('CosyVoice{0} 模型使用 instruct2 模式，请提供 prompt 音频（目标音色）'.format(
                    '3' if model_class == 'CosyVoice3' else '2'))
                yield (cosyvoice.sample_rate, default_data), ''
                return
            if prompt_text != '':
                gr.Info('您正在使用自然语言控制模式，prompt文本会被忽略')
            logging.info('get instruct2 inference request (model: {})'.format(model_class))
            set_all_random_seed(seed)
            chunks = [i['tts_speech'].numpy().flatten()
                      for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav,
                                                              stream=stream, speed=speed)]
            output_path = save_output_audio(chunks, cosyvoice.sample_rate)
            if output_path:
                gr.Info('音频已保存至: {}'.format(output_path))
                yield output_path, output_path
            else:
                yield (cosyvoice.sample_rate, default_data), ''
        else:
            if sft_dropdown == '' or sft_dropdown is None:
                gr.Warning('CosyVoice v1 模型使用 instruct 模式，请选择预训练音色')
                yield (cosyvoice.sample_rate, default_data), ''
                return
            if prompt_wav is not None or prompt_text != '':
                gr.Info('您正在使用自然语言控制模式，prompt音频/prompt文本会被忽略')
            logging.info('get instruct inference request (model: {})'.format(model_class))
            set_all_random_seed(seed)
            chunks = [i['tts_speech'].numpy().flatten()
                      for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text,
                                                              stream=stream, speed=speed)]
            output_path = save_output_audio(chunks, cosyvoice.sample_rate)
            if output_path:
                gr.Info('音频已保存至: {}'.format(output_path))
                yield output_path, output_path
            else:
                yield (cosyvoice.sample_rate, default_data), ''

    # --- 语音转换 (Voice Conversion) ---
    elif mode_checkbox_group == '语音转换':
        if source_wav is None:
            gr.Warning('请上传源音频（提供语音内容）')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        if prompt_wav is None:
            gr.Warning('请上传或录制prompt音频（提供目标音色）')
            yield (cosyvoice.sample_rate, default_data), ''
            return
        logging.info('get vc inference request')
        set_all_random_seed(seed)
        chunks = [i['tts_speech'].numpy().flatten()
                  for i in cosyvoice.inference_vc(source_wav, prompt_wav,
                                                  stream=stream, speed=speed)]
        output_path = save_output_audio(chunks, cosyvoice.sample_rate)
        if output_path:
            gr.Info('音频已保存至: {}'.format(output_path))
            yield output_path, output_path
        else:
            yield (cosyvoice.sample_rate, default_data), ''


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [Fun-CosyVoice3-0.5B](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) \
                    [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(
            label="输入合成文本",
            lines=3,
            value="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            placeholder="输入要合成的文本。支持细粒度控制: [breath] [laughter] <strong>...</strong> 等；支持拼音纠错: [j][ǐ]"
        )
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label='选择推理模式',
                value=inference_mode_list[1]
            )
            instruction_text = gr.Text(
                label="操作步骤",
                value=instruct_dict[inference_mode_list[1]],
                scale=2
            )
            sft_dropdown = gr.Dropdown(
                choices=sft_spk,
                label='选择预训练音色',
                value=sft_spk[0] if sft_spk and sft_spk[0] else None,
                scale=1
            )
            stream = gr.Radio(
                choices=stream_mode_list,
                label='是否流式推理',
                value=stream_mode_list[0][1]
            )
            speed = gr.Number(
                value=1,
                label="速度调节(仅支持非流式推理)",
                minimum=0.5,
                maximum=2.0,
                step=0.1
            )
            with gr.Column(scale=1):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            device_radio = gr.Radio(
                choices=device_mode_list,
                label='推理设备',
                value=device_mode_list[0][1] if torch.cuda.is_available() else device_mode_list[1][1],
                scale=2
            )
            switch_device_button = gr.Button("切换设备", variant="secondary", scale=1)

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload',
                type='filepath',
                label='Prompt 音频文件 / 目标音色音频\n(采样率不低于16kHz)'
            )
            prompt_wav_record = gr.Audio(
                sources='microphone',
                type='filepath',
                label='录制 Prompt 音频'
            )
        source_wav_upload = gr.Audio(
            sources='upload',
            type='filepath',
            label='源音频（仅语音转换模式使用，提供语音内容）'
        )
        prompt_text = gr.Textbox(
            label="输入 prompt 文本",
            lines=2,
            placeholder="请输入 prompt 文本，需与 prompt 音频内容一致。\nCosyVoice3 格式: You are a helpful assistant.<|endofprompt|>音频对应文字",
            value='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        )
        with gr.Row():
            instruct_preset = gr.Dropdown(
                choices=[''] + instruct_preset_choices,
                label='指令预设 (自然语言控制模式)',
                value='',
                scale=1
            )
            instruct_text = gr.Textbox(
                label="输入 instruct 文本",
                lines=2,
                placeholder="请输入 instruct 文本。\nCosyVoice3 格式: You are a helpful assistant. <指令><|endofprompt|>\n可从左侧指令预设下拉框选择",
                value='',
                scale=1
            )

        with gr.Accordion("示例文本（点击自动填充）", open=False):
            gr.Examples(
                examples=[
                    ["八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", "3s极速复刻", ""],
                    ["高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。", "3s极速复刻", ""],
                    ["You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]", "跨语种复刻", ""],
                    ["You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ。", "跨语种复刻", ""],
                    ["好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。", "自然语言控制", "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"],
                    ["收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "自然语言控制", "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>"],
                ],
                inputs=[tts_text, mode_checkbox_group, instruct_text],
            )

        with gr.Row():
            generate_button = gr.Button("生成音频", variant="primary", scale=4)
            stop_button = gr.Button("停止生成", variant="stop", scale=1)

        audio_output = gr.Audio(label="合成音频", autoplay=True)
        file_path_output = gr.Textbox(label="音频文件路径", interactive=False, placeholder="生成后显示文件路径")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        switch_device_button.click(fn=reload_model, inputs=[device_radio], outputs=[sft_dropdown])
        generate_event = generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text,
                                      prompt_wav_upload, prompt_wav_record, source_wav_upload,
                                      instruct_text, seed, stream, speed],
                              outputs=[audio_output, file_path_output])
        stop_button.click(fn=stop_audio_generation, cancels=[generate_event])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        instruct_preset.change(fn=select_instruct_preset, inputs=[instruct_preset], outputs=[instruct_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help='device for inference (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='enable fp16 mixed precision for GPU inference to reduce VRAM usage')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir, device=args.device, fp16=args.fp16)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
