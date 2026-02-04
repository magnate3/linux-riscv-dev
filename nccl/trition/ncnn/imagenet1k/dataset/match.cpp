#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text = "n02395406_hog.JPEG";
    std::regex pattern(R"(^([^_]+)_([^.]+)\.JPEG$)"); // 定义分组
    std::smatch matches;

    // 类似 re.match(pattern, text)
    if (std::regex_match(text, matches, pattern)) {
        std::cout << "Full match: " << matches[0] << std::endl;
        
        std::cout << "Year: " << matches[1] << std::endl;
        std::cout << "Month: " << matches[2] << std::endl;
        std::cout << "Day: " << matches[3] << std::endl;
    } else {
        std::cout << "Match failed" << std::endl;
    }

    return 0;
}
