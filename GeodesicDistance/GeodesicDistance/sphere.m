r = 2.0;
N = 2000;
k = 100;
phi =  5*randn(N,1);
theta = 5*randn(N,1);

error_sigma = 0.1;

x = r*sin(phi).*cos(theta) ;
y = r*sin(phi).*sin(theta) ;
z = r*cos(phi);
idx = z>0;

X = zeros(N,3);
X(:,1) = x;
X(:,2) = y;
X(:,3) = z;

d = 2;
X = X(idx,:) + error_sigma*randn(size(X(idx,:)));

[D,SD] = GeoDist(X,k,d);

[C,I] = max(SD(:));
[I1,I2,I3,I4] = ind2sub(size(SD),I);;

rotate3d on
scatter3(X(:,1),X(:,2),X(:,3),10,'filled', 'b')
hold on 
scatter3(X(I1,1), X(I1,2), X(I1,3), 50, 'filled', 'r')
hold on
scatter3(X(I2,1), X(I2,2), X(I2,3), 50, 'filled', 'r')

SD_1 = zeros(size(SD));
for i = 1:size(SD_1,1)
    for j = 1:size(SD_1,2) 
        SD_1(i,j) = SD(i,j) + log(SD(i,I1)*SD(i,I2)) + log(SD(j,I1)*SD(j,I2));
    end
end

[C_,J] = max(SD_1(:));
[J1,J2,J3,J4] = ind2sub(size(SD_1),J);

hold on 
scatter3(X(J1,1), X(J1,2), X(J1,3), 50, 'filled', 'g')
hold on
scatter3(X(J2,1), X(J2,2), X(J2,3), 50, 'filled', 'g')

dist_1 = zeros(size(X,1),1);
dist_2 = zeros(size(X,1),1);

for i = 1:size(X,1) 
    dist_1(i) = SD(i,I1)/SD(i,I2);
    dist_2(i) = SD(i,J1)/SD(i,J2);
end
    
[c,i] = min(abs(dist_1-1)+abs(dist_2-1))
    
hold on
scatter3(X(i,1), X(i,2), X(i,3), 100, 'filled', 'k')

dlmwrite('sphere.txt',X) 
dlmwrite('pw_dist.txt',SD) 
    