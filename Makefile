tra:
	gcc scalingOMP.cpp -o tra 
	./tra f1.jpg 2.0
ari:
	gcc arithOMP.cpp -o ari 
	./ari f1.jpg f2.jpg 3
gau1:
	gcc convOMP.cpp -o gau 
	./gau sapo.jpg g 15 1
lap:
	gcc convOMP.cpp -o gau 
	./gau sapo.jpg l
sep:
	gcc sepOMP.cpp -o sep 
	./sep sapo.jpg 7 1
knn:
	gcc knnOMP.cpp -o knn 
	./knn sapo.jpg 7 50
col:
	gcc colorOMP.cpp -o col 
	./col sapo.jpg 180