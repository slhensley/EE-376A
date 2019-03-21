import numpy as np
import collections
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import os
import imageio


def get_mc(count):
	mclist = count.most_common()
	largest_num = mclist[0][1]
	largest_vals = [i for (i,j) in mclist if j==largest_num]
	return np.random.choice(largest_vals)


def str2col(count):
	if list(count) == []:
		return '#ffffff'
	else:
		label = get_mc(mem_dict[i])
		num = sum([int(a) for a in label])
		if num==0:
			return '#33ccff'
		elif num==1:
			return '#e6ccff'
		elif num==2:
			return '#cc99ff'
		elif num==3:
			return '#ff99ff'
		elif num==4:
			return '#ff3399'
		elif num==5:
			return '#ff0000'


def generate_graph(edges):
	G = nx.Graph()
	G.add_nodes_from(range(edges.shape[0]))
	for i in range(edges.shape[0]):
		for j in range(i,edges.shape[0]):
			if edges[i,j] == 1:
				G.add_edge(i,j)
	return G


def set_graph_attrs(G):
	A = nx.nx_agraph.to_agraph(G)
	A.node_attr['shape']='circle'
	A.node_attr['fixedsize']='true'
	A.node_attr['fontsize']='7'
	A.node_attr['style']='filled'
	A.node_attr['height']='0.2'
	A.node_attr['width']='0.2'
	A.graph_attr['outputorder']='edgesfirst'
	A.graph_attr['size']='6!,6!'
	A.edge_attr['color']='#000000'
	A.edge_attr['style']='setlinewidth(2)'
	return A


def generate_new_graph(beta,T,m0=5,m=2,p=0.5):
	beta = 1

	repeat = True
	# generate random symmetric matrix
	while repeat:
		repeat = False
		edges = np.random.rand(m0,m0)
		edges = 0.5*edges + 0.5*edges.T
		edges[edges>p] = 1
		edges[edges<=p] = 0
		for i in range(m0):
			edges[i,i] = 0
			if sum(edges[i,:]) == 0:
				repeat = True
	# growth
	for t in range(T):
		prob = [sum(edges[i,:]) for i in range(edges.shape[0])]
		new_edges = np.random.choice(range(edges.shape[0]),m, replace=False,p=prob/sum(prob))
		new_col = np.zeros((edges.shape[0],1))
		new_col[new_edges] = 1
		new_row = np.append(new_col,np.zeros((1,1)),axis=0).T
		edges = np.append(edges,new_col,axis=1)
		edges = np.append(edges,new_row,axis=0)

	eta = np.zeros(edges.shape)
	for i in range(edges.shape[0]):
		for j in range(i,edges.shape[1]):
			eta[i,j] = sum(edges[j,:])**beta/max([sum(edges[k,:])**beta for k in np.flatnonzero(edges[i,:])])
			eta[j,i] = eta[i,j]

	np.save('eta_'+str(m0+T),eta)
	np.save('edges_'+str(m0+T),edges)
	return eta,edges


def run_rumors(init_person,edges,fname,create_gif,K,Hmax,num_rounds,eta,G,A,L,mcr=True,liarnum=0,truthnum=0):
	mem_dict = [collections.Counter() for i in range(edges.shape[0])]
	mem_list = [[] for i in range(edges.shape[0])]

	if create_gif:
		for i in range(edges.shape[0]):
			n = A.get_node(i)
			n.attr['label'] = ''

	liars = np.random.choice([k for k in range(edges.shape[0]) if k!=init_person],size=liarnum,replace=False)
	for l in liars:
		mem_dict[l]['11111'] += 1
		mem_list[l] += ['11111']
		if create_gif:
			n = A.get_node(l)
			n.attr['label'] = 'L'

	truths = np.random.choice([k for k in range(edges.shape[0]) if k not in liars and k!=init_person],size=truthnum,replace=False)
	for t in truths:
		mem_dict[t]['00000'] += 1
		mem_list[t] += ['00000']
		if create_gif:
			n = A.get_node(t)
			n.attr['label'] = 'T'

	mem_dict[init_person]['00000'] += 1
	mem_list[init_person] += ['00000']

	H = [0 for i in range(edges.shape[0])]
	avgH = []
	varH = []
	maxH = []
	minH = []
	opinion_frag = np.zeros((32,num_rounds))

	if create_gif:
		if not os.path.exists(fname+'_images'):
			os.mkdir(fname+'_images')

	for itnum in range(num_rounds):
		updates = [[] for i in range(edges.shape[0])]

		if create_gif:
			A.graph_attr['label'] = 't = '+format(itnum,'03')
			for i in range(edges.shape[0]):
				n = A.get_node(i)
				n.attr['fillcolor'] = str2col(mem_dict[i])
			A.draw(fname+'_images/'+fname+'_'+str(itnum)+'.png',prog='neato')

		for i in range(edges.shape[0]):
			totrum = sum((mem_dict[i]).values())
			if totrum != 0:
				if i in liars or i in truths:
					rumor = get_mc(mem_dict[i])
				else:
					if mcr:
						rumor = get_mc(mem_dict[i])
					else:
						rumor = np.random.choice(list(mem_dict[i]),p=[k/sum(mem_dict[i].values()) for k in mem_dict[i].values()])
					H[i] = sum([- val/totrum*np.log2(val/totrum) for val in (mem_dict[i]).values()])
					Pi = 1/(np.exp((Hmax-H[i])*K/Hmax) + 1)
					mutate = np.random.choice([True,False],p=[Pi,1.0-Pi])
					if mutate:
						bitflip = np.random.choice(range(5))
						new_rumor = rumor[0:bitflip]+str(1-int(rumor[bitflip]))+rumor[bitflip+1:]
						mem_dict[i][new_rumor] += 1
						mem_dict[i][rumor] -= 1
						if mem_dict[i][rumor] == 0:
							del mem_dict[i][rumor]
						for j in range(len(mem_list[i])):
							if mem_list[i][j] == rumor:
								mem_list[i][j] = new_rumor
								break
						rumor = new_rumor
				for j in np.flatnonzero(edges[i,:]):
					if j in liars or j in truths:
						accept = False
					else:
						accept = np.random.choice([True,False],p=[eta[j,i],1.0-eta[j,i]])
					if accept:
						updates[j] += [rumor]

		for j, update_list in enumerate(updates):
			for up in update_list:
				mem_dict[j][up] += 1
				mem_list[j] += [up]
				if len(mem_list[j]) > L:
					first = mem_list[j].pop(0)
					mem_dict[j][first] -= 1
					if mem_dict[j][first] == 0:
						del mem_dict[j][first]

		avgH += [sum(H)/len(H)]
		varH += [sum([(H[i] - avgH[-1])**2/len(H) for i in range(edges.shape[0])])]
		maxH += [max(H)]
		minH += [min(H)]

		current_opinions = collections.Counter([get_mc(mem_dict[i]) for i in range(edges.shape[0]) if list(mem_dict[i])!=[]])
		for opkey,binstr in [(a,format(a,'05b')) for a in range(32)]:
			opinion_frag[opkey,itnum] = current_opinions[binstr]/edges.shape[0]

	if create_gif:
		with imageio.get_writer(fname+'.gif', mode='I',fps=5.0) as writer:
			for filename in [fname+'_images/'+fname+'_'+str(i)+'.png' for i in range(num_rounds)]:
				image = imageio.imread(filename)
				writer.append_data(image)

	return np.array(avgH),np.array(varH),np.array(maxH),np.array(minH),np.array(opinion_frag)


def make_plots(avgH,varH,maxH,minH,opinion_frag,avgH_e,varH_e,maxH_e,minH_e,opinion_frag_e):
	x = range(len(avgH))

	plt.figure(1)
	plt.plot(x,avgH,'tab:blue',label=r'$\bar{H}$')
	plt.plot(x,[avgH[i] + np.sqrt(varH[i]) for i in x],color='tab:blue',linestyle='dotted',label=r'$\pm$ std dev')
	plt.plot(x,[avgH[i] - np.sqrt(varH[i]) for i in x],color='tab:blue',linestyle='dotted',label='_nolegend_')

	plt.plot(x,avgH_e,'tab:orange',label=r"$\bar{H}'$")
	plt.plot(x,[avgH_e[i] + np.sqrt(varH_e[i]) for i in x],color='tab:orange',linestyle='dotted',label=r'$\pm$ std dev')
	plt.plot(x,[avgH_e[i] - np.sqrt(varH_e[i]) for i in x],color='tab:orange',linestyle='dotted',label='_nolabel_')

	plt.plot(maxH,'tab:green',label=r'$\max(H)$')
	plt.plot(maxH_e,'tab:olive',label=r"$\max(H')$")
	plt.plot(minH,'tab:red',label=r'$\min(H)$')
	plt.plot(minH_e,'tab:pink',label=r"$\min(H')$")

	plt.legend()
	#plt.title('Evolution of Entropy')
	plt.xlabel('t')
	plt.ylabel('Entropy (bits)')
	plt.ylim((0,5))


	fig2, ax2 = plt.subplots()
	im = ax2.imshow(opinion_frag,aspect='auto',vmin=0.0,vmax=0.6)

	ax2.set_xticks([0,10,100])
	ax2.set_yticks(range(32))
	ax2.set_xticklabels(['$0$','$10$','$10^2$'])
	ax2.set_yticklabels([format(a,'05b') for a in range(32)])
	plt.ylabel('Opinion Type')
	plt.xlabel('t')
	#plt.title('Opinion Fragmentation for Control')
	cbar2 = ax2.figure.colorbar(im, ax=ax2)
	cbar2.ax.set_ylabel('proportion holding opinion',rotation=-90,va='bottom')


	fig3, ax3 = plt.subplots()
	im = ax3.imshow(opinion_frag_e,aspect='auto',vmin=0.0,vmax=0.6)
	ax3.set_xticks([0,10,100])
	ax3.set_yticks(range(32))
	ax3.set_xticklabels(['$0$','$10$','$10^2$'])
	ax3.set_yticklabels([format(a,'05b') for a in range(32)])
	plt.ylabel('Opinion Type')
	plt.xlabel('t')
	#plt.title('Opinion Fragmentation for Experimental')
	cbar3 = ax3.figure.colorbar(im, ax=ax3)
	cbar3.ax.set_ylabel('proportion holding opinion',rotation=-90,va='bottom')

	plt.show()


def main():
	L = 320
	Hmax = 5

	gen_new_graph = True
	create_gif = False
	num_rounds = 300
	K = 1
	T = 295 #295 #2995
	beta = 1

	overall_avgH = np.zeros(num_rounds)
	overall_varH = np.zeros(num_rounds)
	overall_maxH = np.zeros(num_rounds)
	overall_minH = np.zeros(num_rounds)
	overall_opinion_frag = np.zeros((32,num_rounds))
	overall_avgH_e = np.zeros(num_rounds)
	overall_varH_e = np.zeros(num_rounds)
	overall_maxH_e = np.zeros(num_rounds)
	overall_minH_e = np.zeros(num_rounds)
	overall_opinion_frag_e = np.zeros((32,num_rounds))

	trials = 100
	for it_counter in range(trials):
		print(it_counter)
		#print('initializing graph')
		if gen_new_graph:
			eta,edges = generate_new_graph(beta,T)
		else:
			edges = np.load('edges_55.npy')
			eta = np.load('eta_55.npy')

		# assign a rumor to someone at random to init
		init_person = np.random.choice(range(edges.shape[0]))
		G = generate_graph(edges)
		if create_gif:
			A = set_graph_attrs(G)
		else:
			A = None

		#print('starting control')
		exp_1_name = 'control'
		avgH,varH,maxH,minH,opinion_frag = run_rumors(init_person,edges,exp_1_name,create_gif,K,Hmax,num_rounds,eta,G,A,L)

		#print('starting experimental')
		exp_2_name = 'one_each'
		avgH_e,varH_e,maxH_e,minH_e,opinion_frag_e = run_rumors(init_person,edges,exp_2_name,create_gif,K,Hmax,num_rounds,eta,G,A,L,liarnum=15,truthnum=15)

		#make_plots(avgH,varH,maxH,minH,opinion_frag,avgH_e,varH_e,maxH_e,minH_e,opinion_frag_e)

		overall_avgH += avgH/trials
		overall_varH += varH/trials
		overall_maxH += maxH/trials
		overall_minH += minH/trials
		overall_opinion_frag += opinion_frag/trials
		overall_avgH_e += avgH_e/trials
		overall_varH_e += varH_e/trials
		overall_maxH_e += maxH_e/trials
		overall_minH_e += minH_e/trials
		overall_opinion_frag_e += opinion_frag_e/trials


	make_plots(overall_avgH,overall_varH,overall_maxH,overall_minH,overall_opinion_frag,overall_avgH_e,overall_varH_e,overall_maxH_e,overall_minH_e,overall_opinion_frag_e)



if __name__ == '__main__':
	main()