
#include "dataloader.hpp" 
#include "evaluate.hpp"
#include <string>
#include <fstream>

int main(int argc, char* argv[]){ 
	std::cout << " -- Project: evaluate" << std::endl; 

	std::string source_dir;
	std::string model_name;
	if (argc == 3) {
		model_name = argv[1];
		source_dir = argv[2];
		std::cout << "Using given model " << model_name << " and data path " << source_dir << std::endl;
	} else if (argc == 1) {
		model_name = "squeezenet_v1.1";
		source_dir = "/mnt/data/dataset/imagenet/images_val/";
		std::cout << "Using default model " << model_name << " and default data path " << source_dir << std::endl;
	} else {
		std::cout << "Args count error. " << std::endl;
		std::cout << "./evaluate [model_name] [data_path]" << std::endl;
		return 1;
	}
	// "/mnt/data/dataset/imagenet/images_val/"; 
	evaluate::Evaluate eval(source_dir, model_name);
	eval.process();
	
	return 0; 
}
