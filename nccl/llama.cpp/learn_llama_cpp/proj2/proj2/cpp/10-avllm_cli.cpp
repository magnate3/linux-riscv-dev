/*
 * [text-input] -> [format chatML] -> [generated text]
 *
 */
#include <inttypes.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#ifdef _MSC_VER
#include <ciso646>
#endif

static void print_usage(int, char **argv) {
  printf("\nexample usage:\n");
  printf("\n    %s -m model.gguf [-c ctx]\n", argv[0]);
  printf("\n");
}

int main(int argc, char **argv) {
  // Use a parameter struct similar to llama.cpp
  common_params params;
  if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON,
                           print_usage)) {
    return 1;
  }

  common_init();

  // Backend initialization
  llama_backend_init();
  llama_numa_init(params.numa);

  LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
#if 0
  common_params_context llama_init = common_init_from_params(params);

  llama_model *model = llama_init.model.get();
  llama_context *ctx = llama_init.context.get();

#else
  auto llama_init = common_init_from_params(params);

  auto * model = llama_init->model();
  auto * ctx   = llama_init->context();
#endif
  if (model == nullptr) {
    LOG_ERR("%s: error: unable to load model\n", __func__);
    return -1;
  }
  if (ctx == nullptr) {
    LOG_ERR("%s: error: failed to create the llama_context\n", __func__);
    return -1;
  }

  // Threadpool/Backend (ggml) setup (refactored to match llama.cpp main.cpp)
  auto *cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (!cpu_dev) {
    LOG_ERR("%s: no CPU backend found\n", __func__);
    return 1;
  }
  auto *reg = ggml_backend_dev_backend_reg(cpu_dev);
  auto *ggml_threadpool_new_fn =
      (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_new");
  auto *ggml_threadpool_free_fn =
      (decltype(ggml_threadpool_free) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_free");

  struct ggml_threadpool_params tpp =
      ggml_threadpool_params_from_cpu_params(params.cpuparams);

  struct ggml_threadpool *threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool) {
    LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__,
            tpp.n_threads);
    return 1;
  }

  // Attach threadpool to context
  llama_attach_threadpool(ctx, threadpool, nullptr);

  // ---- Sampling setup refactored to match llama.cpp (main.cpp) ----
  auto &sparams = params.sampling;
  common_sampler *smpl = common_sampler_init(model, sparams);
  if (!smpl) {
    LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
    ggml_threadpool_free_fn(threadpool);
    llama_backend_free();
    return 1;
  }

  LOG_INF("sampler seed: %u\n", common_sampler_get_seed(smpl));
  // LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
  // LOG_INF("sampler chain: %s\n", common_sampler_print(smpl).c_str());
  std::cout << std::endl;

  const char *chat_tmpl = llama_model_chat_template(model, /* name */ nullptr);
  if (chat_tmpl == nullptr) {
    fprintf(stderr, "%s: error: could no accept the template is null\n",
            __func__);
    return -1;
  }
  std::vector<llama_chat_message> chat_messages;
  std::vector<char> chat_message_output(llama_n_ctx(ctx));
  int chat_message_start = 0;
  int chat_message_end = 0;

  while (true) {
    std::string input_msg;
    {  // get input string and apply template

      bool is_sys = chat_messages.size() == 0;
      std::cout << "\n" << (is_sys ? "system >" : "user   >");

      std::getline(std::cin, input_msg);
      if (input_msg.empty()) continue;

      chat_messages.push_back(
          {strdup(is_sys ? "system" : "user"), strdup(input_msg.c_str())});

      int len = llama_chat_apply_template(
          chat_tmpl, chat_messages.data(), chat_messages.size(), true,
          chat_message_output.data(), chat_message_output.size());
      if (len > (int)chat_message_output.size()) {
        chat_message_output.resize(len);
        len = llama_chat_apply_template(
            chat_tmpl, chat_messages.data(), chat_messages.size(), true,
            chat_message_output.data(), chat_message_output.size());
      }

      if (len < 0) {
        fprintf(stderr, "%s: error: failed to apply chat template", __func__);
        return 1;
      }

      chat_message_end = len;
    }
    std::string prompt(chat_message_output.begin() + chat_message_start,
                       chat_message_output.begin() + chat_message_end);

    chat_message_start = chat_message_end;

    llama_token new_token;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
      fprintf(stderr, "%s: failed to get vocal from model \n", __func__);
      exit(-1);
    }

    bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
    int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                          NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(vocab, prompt.data(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), is_first,
                       true) < 0) {
      fprintf(stderr, "%s: failed to tokenize the prompt \n", __func__);
      exit(-1);
    }

    if (false) {  // print token
      std::cout << "\ntokens: \n";
      int cnt = 0;
      for (auto token : prompt_tokens) {
        char buf[120];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
          fprintf(stderr, "%s: error: failed to tokenize \n", __func__);
          exit(-1);
        }
        std::string s(buf, n);
        std::cout << s;
        cnt++;
      }
      std::cout << "end: " << cnt << "\n";
    }

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    while (true) {
      int n_ctx = llama_n_ctx(ctx);
      int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

      if (n_ctx_used + batch.n_tokens > n_ctx) {
        fprintf(stdout, "%s: the context is exceeded. \n", __func__);
        return -1;
      }

      if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        return -1;
      }

      // new_token = llama_sampler_sample(smpl, ctx, -1);
      new_token = common_sampler_sample(smpl, ctx, -1);
      common_sampler_accept(smpl, new_token, /* accept_grammar= */ true);
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }

      char buf[100];
      int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s, failed to convert a token \n", __func__);
        return 0;
      }

      // handle utf-8 character

      std::string out(buf, n);
      printf("%s", out.c_str());
      fflush(stdout);

      batch = llama_batch_get_one(&new_token, 1);
    }
  }

  if (false) {
    printf("\n");
    // llama_perf_sampler_print(smpl);
    common_perf_print(ctx, smpl);
    llama_perf_context_print(ctx);
  }

  // llama_sampler_free(smpl);
  common_sampler_free(smpl);
  ggml_threadpool_free_fn(threadpool);
  llama_free(ctx);
  llama_model_free(model);

  return 0;
}
