import os,sys

for root, dirs, files in os.walk(sys.argv[1]):
  for filename in files:
    if os.path.splitext(root+"/"+filename)[1] == '.mp4' or os.path.splitext(root+"/"+filename)[1] == '.mkv' or os.path.splitext(root+"/"+filename)[1] == '.webm':
      # print(root+filename)
      fname = os.path.splitext(filename)[0]
      os.system("ffmpeg -i " + root+"/"+filename + " -ac 2 -ar 21000 audio/" + fname+".wav")
      os.system("ffmpeg -i " + root+"/"+filename + " -vf fps=1/3.75 -start_number 0 frames/" + fname+"_%03d.jpg -hide_banner")
      os.system("ffmpeg -i audio/" +fname+ ".wav -f  segment -segment_time 3.75 -start_number 0 audio/"+fname+"_%03d.wav")
      os.system("rm audio/"+fname+".wav")
