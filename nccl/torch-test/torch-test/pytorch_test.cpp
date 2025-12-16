// clang++ -std=c++17 -I ~/local/include/torch/csrc/api/include/ -I ~/local/include -L ~/local/lib pytorch.cpp -lc10 -ltorch_cpu -rpath ~/local/lib -o pytorch
// https://pytorch.org/cppdocs/
#include <torch/torch.h>
#include <c10/util/Exception.h>
#include <iostream>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

extern "C" void test_linear_layer_and_mse_loss(
	float *Wgrad, float *Xgrad, float *Bgrad, float *Ygrad, float *TWOgrad
);

// This will be very helpful for some automated testing!
int main() {
	int seed = 42;
	srand(seed);
	torch::manual_seed(seed);
	at::globalContext().setDeterministicAlgorithms(true, true);

	torch::Tensor W, X, B, Y, TWO;

	W = (torch::ones({3, 3})*1).set_requires_grad(true);
	X = (torch::ones({3, 2})*2).set_requires_grad(true);
	B = (torch::ones({3, 2})*3).set_requires_grad(true);
	Y = (torch::ones({3, 2})*4).set_requires_grad(true);
	TWO = (torch::ones({1})*2).set_requires_grad(true);

	torch::Tensor linear = torch::sigmoid((W.matmul(X) + B));
	torch::Tensor L = torch::sum(torch::pow((linear - Y), TWO));
	L.backward();

	std::vector<float> Wgrad(torch::numel(W));
	std::vector<float> Xgrad(torch::numel(X));
	std::vector<float> Bgrad(torch::numel(B));
	std::vector<float> Ygrad(torch::numel(Y));
	std::vector<float> TWOgrad(torch::numel(TWO));

	//test_linear_layer_and_mse_loss(&Wgrad[0], &Xgrad[0], &Bgrad[0], &Ygrad[0], &TWOgrad[0]);

	if (!torch::tensor(Wgrad).reshape_as(W).allclose(W.grad())) {
		std::cerr << "bad W\n";
	}
	if (!torch::tensor(Xgrad).reshape_as(X).allclose(X.grad())) {
		std::cerr << "bad X\n";
	}
	if (!torch::tensor(Bgrad).reshape_as(B).allclose(B.grad())) {
		std::cerr << "bad B\n";
	}
	if (!torch::tensor(Ygrad).reshape_as(Y).allclose(Y.grad())) {
		std::cerr << "bad Y\n";
	}
	float rtol=1e-05; float atol=1e-08; bool equal_nan=true;
	if (!torch::tensor(TWOgrad).reshape_as(TWO).allclose(TWO.grad(), rtol, atol, equal_nan)) {
		std::cerr << TWO.grad() << '\n';
		std::cerr << "bad TWO\n";
	}

	return 0;
}
