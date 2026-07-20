#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "data_loader.hpp"
#include "tokenizer/tokenizer.hpp"

namespace tnn {

class OpenWebTextDataLoader : public BaseDataLoader {
private:
  DType_t dtype_ = DType_t::FP32;
  size_t context_length_;
  int padding_token_id_ = -1;  // -1 means no padding token (default behavior)

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ + batch_size > num_samples_) {
      return false;
    }

    batch_data = Tensor::create<T>({batch_size, context_length_});
    batch_labels = Tensor::create<T>({batch_size, context_length_, vocab_size_});

    T *label_ptr = static_cast<T *>(batch_labels->data());
    std::fill(label_ptr, label_ptr + batch_labels->size(), static_cast<T>(0));

    for (size_t b = 0; b < batch_size; ++b) {
      size_t start_pos;
      if (shuffled_) {
        start_pos = dist_(this->rng_);
        this->current_index_++;
      } else {
        start_pos = this->current_index_++;
      }

      for (size_t i = 0; i < context_length_; ++i) {
        batch_data->at<T>({b, i}) = static_cast<T>(mapped_data_[start_pos + i]);
        int token_id = static_cast<int>(mapped_data_[start_pos + i + 1]);
        // Only set label if token is valid and not a padding token
        if (token_id >= 0 && token_id < (int)vocab_size_ && token_id != padding_token_id_) {
          batch_labels->at<T>({b, i, (size_t)token_id}) = static_cast<T>(1);
        }
      }
    }

    return true;
  }

public:
  OpenWebTextDataLoader(size_t context_length, DType_t dtype = DType_t::FP32,
                        int padding_token_id = -1)
      : dtype_(dtype), context_length_(context_length), padding_token_id_(padding_token_id) {}

  ~OpenWebTextDataLoader() {
    if (mapped_data_ != MAP_FAILED && mapped_data_ != nullptr) {
      munmap(mapped_data_, file_size_);
    }
    if (fd_ != -1) {
      close(fd_);
    }
  }

  bool load_data(const std::string &source) override {
    fd_ = open(source.c_str(), O_RDONLY);
    if (fd_ == -1) {
      perror("Error opening file");
      return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) return false;
    file_size_ = sb.st_size;

    mapped_data_ = (uint16_t *)mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
      perror("mmap failed");
      return false;
    }

    total_tokens_ = file_size_ / sizeof(uint16_t);

    if (total_tokens_ <= context_length_ + 1) return false;
    num_samples_ = total_tokens_ - context_length_ - 1;

    std::cout << "Total tokens in dataset: " << total_tokens_ << std::endl;

    if (!tokenizer_.load("data/open-web-text/vocab.bin")) {
      std::cerr << "Failed to load vocab from: data/open-web-text/vocab.bin" << std::endl;
      return false;
    }
    vocab_size_ = tokenizer_.vocab_size();

    if (num_samples_ > 0) {
      dist_ = std::uniform_int_distribution<size_t>(0, num_samples_ - 1);
    }

    return true;
  }

  bool get_batch(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) override {
    DISPATCH_ON_DTYPE(dtype_, T, return get_batch_impl<T>(batch_size, batch_data, batch_labels));
  }

  void reset() override { this->current_index_ = 0; }

  void shuffle() override { shuffled_ = true; }

  size_t size() const override { return num_samples_; }

  size_t vocab_size() const { return vocab_size_; }

  int padding_token_id() const { return padding_token_id_; }

  std::vector<size_t> get_data_shape() const override { return {context_length_}; }

private:
  int fd_ = -1;
  uint16_t *mapped_data_ = nullptr;
  size_t file_size_ = 0;
  size_t total_tokens_ = 0;
  size_t num_samples_ = 0;
  Tokenizer tokenizer_;
  size_t vocab_size_ = 0;

  bool shuffled_ = false;
  std::uniform_int_distribution<size_t> dist_;
};

}  // namespace tnn