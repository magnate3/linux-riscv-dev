#define IO_PIC     0x20
#define IRQ_OFFSET 32

    .code16gcc
    .text
    .globl _start
    .type  _start, @function

_start:
    xorw %ax, %ax
    xorw %di, %di
    movw %ax, %es

set_pid:
    # ICW1
    mov $0x11, %al
    mov $(IO_PIC), %dx
    out %al, %dx
    # ICW2
    mov $(IRQ_OFFSET), %al
    mov $(IO_PIC+1), %dx
    out %al, %dx
    # ICW3
    mov $0x00, %al
    mov $(IO_PIC+1), %dx
    out %al, %dx
    # ICW4
    mov $0x3, %al
    mov $(IO_PIC+1), %dx
    out %al, %dx

set_uart_handler:
    mov  $(IRQ_OFFSET * 4 + 4 * 4), %di # IRQ_OFFSET*4 + 4*4
    movw $rx_handler, %es:(%di)
    movw %cs, %es:2(%di)

    sti

# wait rx:
loop:
        1:
        pause
        jmp 1b

rx_handler:
    pushaw
    pushfw
    xor %al, %al
    mov $0x3f8, %dx
    in  %dx, %al
    
    out %al, %dx
    popfw
    popaw
    
    iretw
