import pickle
f=open('email_authors.pkl','r')
x=pickle.load(f)
print type(x)
print len(x)
for i in range(1,100):
	print x[i]
