CC :=				gcc

NAME :=				numa
LDFLAGS :=			-lnuma
BUILD := 			build
SRC_DIR := 			src
INC_DIR := 			include

SRC := 				$(wildcard $(SRC_DIR)/*.c)
OBJ := 				$(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o, $(SRC))
DEP := 				$(OBJ:.o=.d)

CFLAGS := 			-g3 -I$(INC_DIR)

all: $(NAME)

$(NAME): $(OBJ)
	@echo "Compiling $@..."
	$(CC) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: $(SRC_DIR)/%.c | $(BUILD)
	$(CC) -MMD -MF $(BUILD)/$*.d -c $< -o $@ $(CFLAGS)

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)/*.o
	rm -rf $(BUILD)/*.d

fclean: clean
	rm -rf $(NAME)

re: fclean all

-include $(DEP)

.PHONY: all clean fclean
