from pylab import *
import numpy as np

def read_content(fpath):
	data = [{}] * 100
	data2 = [{}] * 100
	names = []
	times = []
	names2 = []
	times2 = []
	i = 0
	with open(fpath) as f:
		content = f.readlines()

	for line in content:
		if line[:2] == "..":
			if i < 4:
				data[i] = {'name':line[15:-1], 'threads':{}}
			else:
				data2[i-4] = {'name':line[33:-1]+'CO', 'threads':{}}
			i+=1
		elif line[0] == "#":
			pos = line.find(":")
			posEnd = line.find(",")
			n = int(line[pos+2:posEnd])
			pos = line.find(":", posEnd)
			time = float(line[pos+2:])
			if i <= 4:
				data[i-1]['threads'][n] = time
				names.append(data[i-1]['name']+", "+str(n))
				times.append(time)
			else:
				data2[i-5]['threads'][n] = time
				names2.append(data2[i-5]['name']+", "+str(n))
				times2.append(time)
	return names, times, names2,times2


def make_plot(names, times, times2):
	figure(1)
	x = range(len(names))
	xticks(x, names, rotation='vertical')
	margins(0.3)
	subplots_adjust(bottom=0.3)
	plot(x, times, '-ro', label='without code optimization')
	plot(x, times2, '-go', label='with code optimization')
	xlabel('Tests')
	ylabel('Time(s)')
	title('Corei5')
	legend(loc='upper right')
	savefig('test.pdf')
	show()
	#savefig('../Doc/Images/corei5.pdf')

def make_plot2(names, times, tlt):
	figure()
	x = [1,2,3,4,8,16]
	y = [times[0] for i in range(6)]
	plot(x, y, '-ko', label='Sequential')
	plot(x, times[7:13], '-ro', label='Parallel i')
	plot(x, times[1:7], '-go', label='Parallel j')
	plot(x, times[13:19], '-bo', label='Parallel k')
	xlim(0,17)
	ylim(0.3, 0.65)
	xlabel('Number of threads')
	ylabel('Time(s)')
	title(tlt)
	legend(loc='center right')
	savefig(tlt+'.pdf')


if __name__ == "__main__":
	n,t,n2, t2 = read_content("../Results/corei5.txt")
	#make_plot(n,t,t2)
	make_plot2(n,t,'Corei5-JustOpenMP-CloseUp')
	make_plot2(n,t2,'Corei5-CodeOptimization-CloseUp')
