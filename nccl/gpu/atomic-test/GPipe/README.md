# GPipe
Inter communication between GPU processes invoked from different CPU processes using the new low level NVIDIA API.

# Definitions
- Three element gpipe - a simple solution, for performance comparcence. The solution uses Linux named pipe to pass communication through the CPU
- Naive solution - communication between GPU processec invoked from the same CPU process 

# API


    // The GPipe constructor
    // Parameters:
    // - pipe_name - The GPipe name. must be unique per pipe instance
    // - is_consumer - Is the caller a consumer (Consumer or Producer)
    // - size_multiplier – Max number of messages per thread in the buffer
    GPipe(const char* pipe_name, bool is_consumer, int size_multiplier);
    
    // The GPipe destructor
    // Close the communication and perform cleanup
    void gclose();
    
    // Initialize the pipe. 
    // Must be called from both pipe ends
    void init();
    
    // Read message from pipe
    // Parameters:
    // - message - Pointer to the reader's buffer
    // Copies the message to the reader's buffer. All the kernel 
    // thread must call this function to read the messages.
    // Returns only when all the kernel threads receive a message
    void gread(message_t* message);
    
    
    // Write message to pipe
    // Parameters:
    // - message - Pointer to the message
    // Copies the message to the pipe buffer.
    // Blocking until the buffer has a writing slot
    void gwrite(message_t* message);


