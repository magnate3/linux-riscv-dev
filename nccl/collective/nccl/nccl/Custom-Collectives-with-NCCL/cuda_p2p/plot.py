import matplotlib.pyplot as plt 
  
  
  
num_plots = 4
  
for i in range(num_plots):
    names = [] 
    marks = [] 

f = open('out_mpi.out','r') 
for row in f: 
    row = row.split(' ') 
    names.append(row[0]) 
    marks.append(int(row[1])) 
  
plt.bar(names, marks, color = 'g', label = 'File Data') 
  
plt.xlabel('Student Names', fontsize = 12) 
plt.ylabel('Marks', fontsize = 12) 
  
plt.title('Students Marks', fontsize = 20) 
plt.legend() 
plt.show() 
