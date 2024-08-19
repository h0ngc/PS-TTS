import argparse
import subprocess
import os

def convert_to_mp4(input_video, output_video):
    if not input_video.endswith('.mp4'):
        subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video, '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', output_video, '-y'])
        input_video = output_video
    return input_video

def extract_and_convert_audio(input_video, output_audio):
    print('1')
    subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', output_audio, '-y'])
    print('2')


def extract_video(input_video, output_video):
    subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video, '-c:v', 'copy', '-an', output_video, '-y'])

def main():
    parser = argparse.ArgumentParser(description='Video and Audio Processing Script')
    parser.add_argument('-i','--input_video', type=str, required=True, default='./input_video/input_video.mp4',help='Path to the input video file')
    parser.add_argument('-o','--output_audio', type=str, required=True, default='./prepare_data/output_audio.wav', help='Path to the output audio file (.wav)')
    parser.add_argument('-v','--output_video', type=str, required=True, default='./prepare_data/output_video.mp4',help='Path to the output video file (.mp4)')

    args = parser.parse_args()

    converted_video = convert_to_mp4(args.input_video, args.output_video)

    extract_and_convert_audio(converted_video, args.output_audio)
    print(args.output_audio)
    extract_video(converted_video, args.output_video)
    

if __name__ == "__main__":
    main()

