import numpy as np
import concurrent.futures
class Custom:
	def __init__(self,head=5):
		self.head = head

	@classmethod
	def Calc(self,point,X,y):
		lst_Dist = []
		for x_point,y_ans in zip(X,y):
			lst_Dist.append([sum((point-x_point)**2) , y_ans])
		return lst_Dist			

	def fit(self,X,y):
		self._X = X.astype(np.int64)
		self._y = y

	def predictPoint(self,point):
		lst_Dist = []

		with concurrent.futures.ThreadPoolExecutor() as executor:
			fut = executor.submit(self.Calc,point,self._X[:2500,:],self._y[:2500])
			fut1 = executor.submit(self.Calc,point,self._X[2501:5000,:],self._y[2501:5000])
			fut2 = executor.submit(self.Calc,point,self._X[5001:7500,:],self._y[5001:7500])
			fut3 = executor.submit(self.Calc,point,self._X[7501:10000,:],self._y[7501:10000])
			
			lst_Dist = fut.result() + fut1.result()+ fut2.result()+ fut3.result()

		lst_Dist.sort(key=lambda x : x[0])
		top = np.array(lst_Dist[:self.head])
		items , count = np.unique(top[:,1],return_counts=True)

		return items[np.argmax(count)]

	def predict(self,X):
		result = []
		for pt in X:
			result.append(self.predictPoint(pt))
		
		return np.array(result,dtype=int)					
	
	def score(self,X,y):
		return sum(self.predict(X)==y)/len(y)


model = Custom(5)
data = np.genfromtxt('mnist_test.csv',delimiter=',')
X = data[:,1:]
y = data[:,0]

print(X.shape , y.shape)

model.fit(X,y)
print(model.predict(X[:30,:]))
print(y[:30])

print(model.score(X[:30,:],y[:30]))
