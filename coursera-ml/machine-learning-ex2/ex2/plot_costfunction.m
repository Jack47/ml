data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

ls = linspace(-1, 1, 80);
v1 = zeros(length(ls), length(ls));
v2 = zeros(size(v1));
d = 2 * length(y);

for i = 1:length(ls)
    for j = 1:length(ls)
        v1(i, j) = sum((sigmoid(X*[ls(i); ls(j)])-y).^2) / d;
        v2(i, j) = costFunction([ls(i); ls(j)], X, y);
    end
end

surf(ls, ls, v1');
figure;
surf(ls, ls, v2');
