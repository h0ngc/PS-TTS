
import subprocess
import argparse
def scp_transfer(src, dst, ip, username, password):
    command = f"sshpass -p {password} scp {src} {username}@{ip}:{dst}"
    subprocess.call(command, shell=True)
    print('scp 완료?')

def merge_audio_video(audio_file, video_file, output_file):
    subprocess.call(['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_file, '-y'])

parser = argparse.ArgumentParser(description='오디오와 비디오 파일 병합 및 전송 스크립트')
parser.add_argument('-af','--audio_file', type=str, required=True, help='오디오 파일 경로')
parser.add_argument('-vf','--video_file', type=str, required=True, help='비디오 파일 경로')
parser.add_argument('-of','--output_file', type=str, required=True, help='병합된 파일을 저장할 경로')
args = parser.parse_args()


merge_audio_video(args.audio_file, args.video_file, args.output_file)

scp_transfer(args.output_file, '/home/server/uploadfile/api_return/mp4','101.101.210.35', 'root', 'F3ND4brg5b32f')

