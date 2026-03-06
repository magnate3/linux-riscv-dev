```
 llama_batch_print
 llama_memory_breakdown_print(ctx); // goes to debug log
```
+   llama_batch_print   
```
static void llama_batch_print(const llama_batch *batch) {
  printf("%s\n", std::string(20, '-').c_str());
  printf("%30s: %-10d\n", "n_tokens", batch->n_tokens);

  printf("tokens|emb:\n");
  for (int i = 0; i < batch->n_tokens && batch->token; i++)
    printf("%8d,", batch->token[i]);
  for (int i = 0; i < batch->n_tokens && batch->embd; i++)
    printf("%8f,", batch->embd[i]);
  printf("\npos: \n");
  for (int i = 0; i < batch->n_tokens && batch->pos; i++)
    printf("%8d,", batch->pos[i]);
  printf("\nn_seq\n");
  for (int i = 0; i < batch->n_tokens && batch->seq_id[i]; i++)
    printf("%8d,", batch->seq_id[i][0]);

  printf("\n");
};
```

```
const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}
```