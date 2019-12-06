import os
import shutil
import codecs


files = os.listdir("/home/priyanka/Desktop/Summarization/dataset_processing/")

for dir_name in files:
	if dir_name.endswith(".zip"):
		print(dir_name)

#exit()

#Creating folders in target location
path = '10301_10400'
limit = path.split('_')
if(len(limit)>1):
    start = int(limit[0])
    end = int(limit[1])
    for i in range(start,end+1):
        fullPath = path+"/"+str(i)
        try:
            os.mkdir(fullPath)
        except OSError:
            print ("Creation of the directory %s failed" % fullPath)
        else:
            print ("Scriptsuccessfully created the directory %s" % fullPath)
    #print("start point {} end point {}",format(start),format(end))
else:
    print("Invalid path given")

#copying files to folders in target location
path = 'dataset/'
copyPath = '10301_10400'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

for f in files:
    temp = f.split('/')
    if(len(temp)>1):
        temp = temp[1].split('.')
        if(len(temp)>1):
            original = copyPath +"/"+ temp[0]
            shutil.copy(f,original);

            originalFile = original + "/" + temp[0] + "1.txt"
            shutil.copy(f,originalFile);
            checked = original +"/"+ temp[0] + ".checked.txt"
            os.rename(originalFile, checked)
            
            sent = original +"/"+ temp[0] + ".sent.txt"
            f = codecs.open(sent, "w", "utf-8")
            f.close()
            summ = original +"/"+ temp[0] + ".summ.txt"
            f = codecs.open(summ, "w", "utf-8")
            f.close()
            summSent = original +"/"+ temp[0] + ".summ.sent.txt"
            f = codecs.open(summSent, "w", "utf-8")
            f.close()
            title = original +"/"+ temp[0] + ".title.txt"
            f = codecs.open(title, "w", "utf-8")
            f.close()

            #print("Original {}",format(original))
            #print("Original {}",format(originalFile))
            #print("Checked {}",format(checked))
            #print("Sentence {}",format(sent))
            #print("Summarization {}",format(summ))
            #print("Sent summarization {}",format(summSent))
            #print("Title {}",format(title))
            #print("\n\n")
            
            #print(temp[0])
        else:
            print("file {} is not valid",format(f))
    else:
        print("file {} is not valid",format(f))

