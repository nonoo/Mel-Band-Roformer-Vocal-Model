#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import time
from pathlib import Path
import sys
import os
import tempfile

import warnings
warnings.filterwarnings("ignore")

SUPPORTED_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac')
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'config_vocals_mel_band_roformer.yaml'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'MelBandRoformer.ckpt'


def collect_input_files(args):
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f'Input file not found: {input_path}')
        return [str(input_path)]

    if not args.input_folder:
        raise ValueError('Either --input or --input_folder must be provided')

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        raise NotADirectoryError(f'Input folder not found: {input_folder}')

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(str(path) for path in input_folder.glob(f'*{ext}'))
        files.extend(str(path) for path in input_folder.glob(f'*{ext.upper()}'))

    return sorted(files)


def read_audio(path):
    import soundfile as sf
    import librosa
    try:
        audio, sample_rate = sf.read(path, always_2d=True)
        return audio.astype(np.float32), sample_rate
    except (RuntimeError, sf.LibsndfileError):
        audio, sample_rate = librosa.load(path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.T.astype(np.float32)
        return audio, sample_rate


def write_f32le_to_stdout(audio):
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    stream_data = np.asarray(audio, dtype='<f4', order='C').tobytes()
    fd = sys.stdout.fileno()
    offset = 0
    while offset < len(stream_data):
        written = os.write(fd, stream_data[offset:])
        if written <= 0:
            raise RuntimeError('Failed to write stream output to stdout')
        offset += written


def write_flac_to_stdout(audio, sample_rate):
    import soundfile as sf
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sample_rate, format='FLAC', subtype='PCM_24')
        with open(tmp_path, 'rb') as f:
            data = f.read()
        fd = sys.stdout.fileno()
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise RuntimeError('Failed to write FLAC stream output to stdout')
            offset += written
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_mp3_to_stdout(audio, sample_rate):
    import soundfile as sf
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sample_rate, bitrate_mode='CONSTANT', compression_level=0.0)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        fd = sys.stdout.fileno()
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise RuntimeError('Failed to write MP3 stream output to stdout')
            offset += written
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def run_inference(model, args, config, device, demix_track, verbose=False):
    import torch
    import soundfile as sf
    from tqdm import tqdm
    start_time = time.time()

    model.eval()
    all_mixtures_path = collect_input_files(args)
    total_tracks = len(all_mixtures_path)
    print('Total tracks found: {}'.format(total_tracks), file=sys.stderr)

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    streaming = args.stream_f32le_instrumental or args.stream_f32le_vocal

    if total_tracks == 0:
        raise ValueError('No input tracks found')

    if streaming and total_tracks != 1:
        raise ValueError('Streaming modes require exactly one input track. Use --input for streaming.')

    if not streaming:
        if not args.store_dir:
            raise ValueError('--store_dir is required when not streaming')
        os.makedirs(args.store_dir, exist_ok=True)

    if not verbose and not streaming:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.basename(path)}", file=sys.stderr)

        mix, sr = read_audio(path)
        original_mono = mix.shape[1] == 1
        if original_mono:
            mix = np.repeat(mix, repeats=2, axis=1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if not streaming and first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds", file=sys.stderr)
            sys.stderr.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stderr.flush()

        stream_callback = None
        if streaming and not args.flac and not args.mp3:
            stream_state = {'cursor': 0}
            instr_key = instruments[0]

            def stream_callback(chunk_dict, output_start, output_end):
                vocals_chunk = chunk_dict[instr_key].T
                if original_mono:
                    vocals_chunk = vocals_chunk[:, 0]
                    mix_slice = mix[output_start:output_end, 0]
                else:
                    mix_slice = mix[output_start:output_end]

                if args.stream_f32le_vocal:
                    write_f32le_to_stdout(vocals_chunk)
                else:
                    instrumental_chunk = mix_slice - vocals_chunk
                    write_f32le_to_stdout(instrumental_chunk)

                stream_state['cursor'] = output_end

        res, first_chunk_time = demix_track(
            config,
            model,
            mixture,
            device,
            first_chunk_time,
            show_progress=not streaming,
            stream_callback=stream_callback
        )

        vocals_output = res[instruments[0]].T
        if original_mono:
            vocals_output = vocals_output[:, 0]

        original_mix = mix[:, 0] if original_mono else mix
        instrumental = original_mix - vocals_output

        if streaming:
            if args.flac:
                if args.stream_f32le_vocal:
                    write_flac_to_stdout(vocals_output, sr)
                else:
                    write_flac_to_stdout(instrumental, sr)
            elif args.mp3:
                if args.stream_f32le_vocal:
                    write_mp3_to_stdout(vocals_output, sr)
                else:
                    write_mp3_to_stdout(instrumental, sr)
            continue

        if args.stream_f32le_vocal:
            write_f32le_to_stdout(vocals_output)
            continue

        if args.stream_f32le_instrumental:
            write_f32le_to_stdout(instrumental)
            continue

        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"

        for instr in instruments:
            current_vocals = res[instr].T
            if original_mono:
                current_vocals = current_vocals[:, 0]

            file_stem = Path(path).stem
            vocals_path = f"{args.store_dir}/{file_stem}_{instr}.{ext}"
            if args.mp3:
                sf.write(vocals_path, current_vocals, sr, bitrate_mode='CONSTANT', compression_level=0.0)
            elif args.flac:
                sf.write(vocals_path, current_vocals, sr, format='FLAC', subtype='PCM_24')
            else:
                sf.write(vocals_path, current_vocals, sr, subtype='FLOAT')

        file_stem = Path(path).stem
        instrumental_path = f"{args.store_dir}/{file_stem}_instrumental.{ext}"
        if args.mp3:
            sf.write(instrumental_path, instrumental, sr, bitrate_mode='CONSTANT', compression_level=0.0)
        elif args.flac:
            sf.write(instrumental_path, instrumental, sr, format='FLAC', subtype='PCM_24')
        else:
            sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time), file=sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Location of the model")
    parser.add_argument("--input", type=str, help="single audio file to process")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action='store_true', help="write/store outputs as FLAC instead of WAV")
    format_group.add_argument("--mp3", action='store_true', help="write/store outputs as MP3 instead of WAV")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream-f32le-instrumental", action='store_true', help='write instrumental to stdout as f32le')
    stream_group.add_argument("--stream-f32le-vocal", action='store_true', help='write vocal to stdout as f32le')
    return parser


def proc_folder(args):
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    import torch
    import torch.nn as nn
    from ml_collections import ConfigDict
    from utils import demix_track, get_model_from_config

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    print('Using model: {}'.format(args.model_path), file=sys.stderr)
    model.load_state_dict(
        torch.load(args.model_path, map_location=torch.device('cpu'))
    )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...', file=sys.stderr)
        model = model.to(device)

    run_inference(model, args, config, device, demix_track, verbose=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        build_parser().print_help()
        sys.exit(0)
    proc_folder(None)
