
```
#include <stdio.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

__u32 test_data32 = 0;
__u32 test1_result = 0;

SEC("sockops")
int bpf_sockmap(struct bpf_sock_ops *skops)
{
        test1_result = __sync_fetch_and_add(&test_data32, 1);
        bpf_printk("test");
        return 0;
}

char _license[] SEC("license") = "GPL";
int _version SEC("version") = 1;
```

编译出错   
```
root@ubuntux86:# clang -target bpf -D__x86_64__ -O2 -c test.c -mcpu=v3   
fatal error: error in backend: Invalid usage of the XADD return value
Stack dump:
0.      Program arguments: clang -target bpf -D__x86_64__ -O2 -c test.c -mcpu=v3 
1.      <eof> parser at end of file
2.      Code generation
3.      Running pass 'Function Pass Manager' on module 'test.c'.
4.      Running pass 'BPF PreEmit Checking' on function '@bpf_sockmap'
 #0 0x00007f17a24924ff llvm::sys::PrintStackTrace(llvm::raw_ostream&) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x9814ff)
 #1 0x00007f17a24907b0 llvm::sys::RunSignalHandlers() (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x97f7b0)
 #2 0x00007f17a2491c4d llvm::sys::CleanupOnSignal(unsigned long) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x980c4d)
 #3 0x00007f17a23e7cea (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x8d6cea)
 #4 0x00007f17a23e7c8b (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x8d6c8b)
 #5 0x00007f17a248d3fe (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x97c3fe)
 #6 0x00000000004125b2 (/usr/lib/llvm-10/bin/clang+0x4125b2)
 #7 0x00007f17a23f19b4 llvm::report_fatal_error(llvm::Twine const&, bool) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x8e09b4)
 #8 0x00007f17a23f18af (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x8e08af)
 #9 0x00007f17a3c98494 (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x2187494)
#10 0x00007f17a272d5e8 llvm::MachineFunctionPass::runOnFunction(llvm::Function&) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0xc1c5e8)
#11 0x00007f17a2597d76 llvm::FPPassManager::runOnFunction(llvm::Function&) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0xa86d76)
#12 0x00007f17a2597ff3 llvm::FPPassManager::runOnModule(llvm::Module&) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0xa86ff3)
#13 0x00007f17a25984a0 llvm::legacy::PassManagerImpl::run(llvm::Module&) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0xa874a0)
#14 0x00007f17a758d433 clang::EmitBackendOutput(clang::DiagnosticsEngine&, clang::HeaderSearchOptions const&, clang::CodeGenOptions const&, clang::TargetOptions const&, clang::LangOptions const&, llvm::DataLayout const&, llvm::Module*, clang::BackendAction, std::unique_ptr<llvm::raw_pwrite_stream, std::default_delete<llvm::raw_pwrite_stream> >) (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x13e8433)
#15 0x00007f17a780ce1c (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x1667e1c)
#16 0x00007f17a69fac13 clang::ParseAST(clang::Sema&, bool, bool) (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x855c13)
#17 0x00007f17a7e70e58 clang::FrontendAction::Execute() (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x1ccbe58)
#18 0x00007f17a7e298a1 clang::CompilerInstance::ExecuteAction(clang::FrontendAction&) (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x1c848a1)
#19 0x00007f17a7ed4daf clang::ExecuteCompilerInvocation(clang::CompilerInstance*) (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x1d2fdaf)
#20 0x000000000041229d cc1_main(llvm::ArrayRef<char const*>, char const*, void*) (/usr/lib/llvm-10/bin/clang+0x41229d)
#21 0x00000000004105b1 (/usr/lib/llvm-10/bin/clang+0x4105b1)
#22 0x00007f17a7b7a8f2 (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x19d58f2)
#23 0x00007f17a23e7c67 llvm::CrashRecoveryContext::RunSafely(llvm::function_ref<void ()>) (/lib/x86_64-linux-gnu/libLLVM-10.so.1+0x8d6c67)
#24 0x00007f17a7b79e2f clang::driver::CC1Command::Execute(llvm::ArrayRef<llvm::Optional<llvm::StringRef> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >*, bool*) const (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x19d4e2f)
#25 0x00007f17a7b5252f clang::driver::Compilation::ExecuteCommand(clang::driver::Command const&, clang::driver::Command const*&) const (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x19ad52f)
#26 0x00007f17a7b526da clang::driver::Compilation::ExecuteJobs(clang::driver::JobList const&, llvm::SmallVectorImpl<std::pair<int, clang::driver::Command const*> >&) const (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x19ad6da)
#27 0x00007f17a7b6593c clang::driver::Driver::ExecuteCompilation(clang::driver::Compilation&, llvm::SmallVectorImpl<std::pair<int, clang::driver::Command const*> >&) (/lib/x86_64-linux-gnu/libclang-cpp.so.10+0x19c093c)
#28 0x000000000041002f main (/usr/lib/llvm-10/bin/clang+0x41002f)
#29 0x00007f17a15f5083 __libc_start_main /build/glibc-wuryBv/glibc-2.31/csu/../csu/libc-start.c:342:3
#30 0x000000000040d7ce _start (/usr/lib/llvm-10/bin/clang+0x40d7ce)
root@ubuntux86:# 
```

当前5.10内核无法支持ebpf中的fetch_and_add的全部能力，issue先行关闭，内核更新后再尝试此功能