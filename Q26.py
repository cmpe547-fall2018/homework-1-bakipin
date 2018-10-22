import numpy as np

x_values = np.array([1, 2])
y_values = np.array([-1, 0, 5])

prob_xy = np.array([[0.3, 0.3, 0], [0.1, 0.2, 0.1]])

marginal_x = np.sum(prob_xy, axis=1).flatten()
marginal_y = np.sum(prob_xy, axis=0).flatten()

expectation_x = np.sum(np.multiply(x_values, marginal_x)).round(decimals=2)
print("<x> : {}".format(expectation_x))

expectation_y = np.sum(np.multiply(y_values, marginal_y)).round(decimals=2)
print("<y> : {}".format(expectation_y))

y_given_x = (prob_xy / marginal_x[:,np.newaxis]).round(decimals=2)
row,col = y_given_x.shape
expectation_yx = np.zeros([row,1])
for i in range(row):
	for j in range(col):
		expectation_yx[i] += y_given_x[i][j]*y_values[j]
expectation_yx = expectation_yx.flatten()
print("<y|x> : {}".format(expectation_yx))

x_given_y = (prob_xy.T / marginal_y[:,np.newaxis]).round(decimals=2)
row,col = x_given_y.shape
expectation_xy = np.zeros([row,1])
for i in range(row):
	for j in range(col):
		expectation_xy[i] += x_given_y[i][j]*x_values[j]
expectation_xy = expectation_xy.flatten()
print("<x|y> : {}".format(expectation_xy))

cov_xy = 0
for i in range(len(x_values)):
	for j in range(len(y_values)):
		cov_xy += prob_xy[i][j]*(x_values[i]-expectation_x)*(y_values[j]-expectation_y)

print("Cov[x,y] : {}".format(cov_xy.round(decimals=2)))

joint_ent = 0
row, col = prob_xy.shape
for i in range(row):
	for j in range(col):
		if prob_xy[i][j] == 0:
			continue
		joint_ent += prob_xy[i][j]*np.log(prob_xy[i][j])

joint_ent = (-joint_ent).round(decimals=2)
print("H[x,y] : {}".format(joint_ent))

ent_x = 0
for i in range(len(marginal_x)):
	if marginal_x[i] == 0:
		continue
	ent_x += marginal_x[i]*np.log(marginal_x[i])

ent_x = (-ent_x).round(decimals=2)
print("H[x] : {}".format(ent_x))	

ent_y = 0
for i in range(len(marginal_y)):
	if marginal_y[i] == 0:
		continue
	ent_y += marginal_y[i]*np.log(marginal_y[i])

ent_y = (-ent_y).round(decimals=2)
print("H[y] : {}".format(ent_y))

ent_yx = 0
row, col = prob_xy.shape
for i in range(row):
	for j in range(col):
		if prob_xy[i][j] == 0:
			continue
		ent_yx += prob_xy[i][j]*np.log(y_given_x[i][j])

ent_yx = (-ent_yx).round(decimals=2)
print("H[y|x] : {}".format(ent_yx))

x_given_y = x_given_y.T
ent_xy = 0
row, col = prob_xy.shape
for i in range(row):
	for j in range(col):
		if prob_xy[i][j] == 0:
			continue
		ent_xy += prob_xy[i][j]*np.log(x_given_y[i][j])

ent_xy = (-ent_xy).round(decimals=2)
print("H[x|y] : {}".format(ent_xy))

mut_info = ent_x - ent_xy
print("I(x,y) : {}".format(mut_info.round(decimals=2)))

print("H(X,Y) != H(X)+H(Y) : {} != {}".format(joint_ent, ((ent_x+ent_y).round(decimals=2))))
print("H(X,Y) == H(X|Y)+H(Y|X)+I(X,Y) : {} == {}".format(joint_ent, ((ent_xy+ent_yx+mut_info).round(decimals=2))))
print("H(Y) == H(Y|X)+I(X,Y) : {} == {}".format(ent_y, ((ent_yx+mut_info).round(decimals=2))))
print("H(X) == H(X|Y)+I(X,Y) : {} == {}".format(ent_x, ((ent_xy+mut_info).round(decimals=2))))
