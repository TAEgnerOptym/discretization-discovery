#function convert_solomon(file_in,file_out,num_cust_keep,divisor,divisor_time_dist)
import numpy as np
import csv
#import pandas as pd

def convert_solomon(file_in,file_out,num_cust_keep,divisor,divisor_time_dist):

	#fileID = fopen(file_in,'r');
	fileID = open(file_in, "r")

	M=[];
	D=[];

	M=fileID.readlines()
	#print(M)

	capacity=M[4]
	capacity=M[4].split(' ')
	capacity=capacity[-1]
	capacity=capacity[0:-1]
	#print('capacity')
	#print(capacity)
	#print('***')


	depot_row=M[9].split(' ');
	my_keep=[]
	for i in range(0,len(depot_row)):
		#print(depot_row[i])
		#print('KK'+str(len(depot_row[i])))
		if len(depot_row[i])>0.5 :
			my_keep.append(int(depot_row[i]))
		if len(my_keep)==7:
			break
	#print('my_keep')
	#print(my_keep)
	#print('my_keep')
	#print('depot_row')
	#print(depot_row)
	D=[my_keep[1]]+[my_keep[2]]+[int(capacity)]+[my_keep[5]]+[0]+[100]
	#print('D')
	#print(D)
	#D=[depot_row(2:3),capacity,depot_row(6),0,100];

	Z=np.zeros((num_cust_keep+1,6))
	Z[0,:]=D
	start_pos=10
	t0=D[3];
	#print(num_cust_keep)
	for i in range(0,num_cust_keep):
		#print("i:  "+ str(i))
		#print(M[i+start_pos])
		my_split=M[i+start_pos].split(' ')
		count=0;
		for j in range(0,len(my_split)):
			if len(my_split[j])>0.5:
				if(count==0):
					count=count+1;
					continue
				#print('len(my_split[j])')
				#print(len(my_split[j]))
				#print(my_split[j])
				Z[i+1,count-1]=int(my_split[j])
				#print(int(my_split[j]))
				count=count+1;
				if(count==7):
					break
		#new_row([4,5])=t0-new_row([4,5]);
		#print('****')
		#print(i)
		#print(Z[i+1,:])
		Z[i+1,3]=t0-Z[i+1,3];
		Z[i+1,4]=t0-Z[i+1,4];
		#print(Z[i+1,:])
		#print(Z[i,:])

	#	D(:,3)=D(:,3)/divisor;

	#print('before')
	#print(Z[:,2])
	Z[:,2]=Z[:,2]/divisor;
	#print('after')
	#print(Z[:,2])

	#D(:,[1,2,4,5,6])=D(:,[1,2,4,5,6])/divisor_time_dist;
	Z[:,[0,1,3,4,5]]=Z[:,[0,1,3,4,5]]/divisor_time_dist;
	#D_old=D;
	D=np.ceil(D);
	#print(Z)
	#print(Z.shape)
	#input('jkn')

	#if(max(abs(D-D_old))>0.5)
	#	disp('odd if this is supposeed to preserve')
	#	pause
	#end

	#csvwrite(file_out,D);
	#print('Z')
	#print(Z)


	fileID.close()

	np.savetxt(file_out, Z, delimiter=",",fmt='%f')


prefix_list=["C1","C2","R1","R2","RC1","RC2"]

for p in prefix_list:
	for i in range(1,10):
		file_in="homberger_200_customer_instances/"+p+"_2_"+str(i)+".TXT";
		file_out="dataHOM/jyHOM_"+p+"_2_"+str(i)+".txt"
		convert_solomon(file_in,file_out,200,1,1)
		#nk=input('---')
	