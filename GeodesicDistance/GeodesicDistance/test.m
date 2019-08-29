n = 1000
k = 50
d = 1

x = 5.5*rand(n,1)
y = zeros(n,2)

% y(:,1) = abs(x).*sin(x) 
% y(:,2) = abs(x).*cos(x)
y(:,1) = 5*sin(x)
y(:,2) = 5*cos(x)

y = y + 0.2*randn(n,2)
scatter(y(:,1), y(:,2),5)


[D,SD] = GeoDist(y,k,d);

sum(sum(abs(D-D')))
sum(sum(abs(SD-SD')))

[C,I] = max(SD(:))
[I1,I2,I3,I4] = ind2sub(size(SD),I);
SD(I1,I2,I3,I4)

scatter(y(:,1), y(:,2),5)
hold on
plot(y(I1,1),y(I1,2),'r*')
plot(y(I2,1),y(I2,2),'r*')

