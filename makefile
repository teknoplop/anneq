CC=clang
OBJ=lms.o lms_init.o eq_funcs.o conio.o
DEP=$(OBJ:%.o=%.d)

CFLAGS=-std=c89

lms.exe: $(OBJ)
	$(CC) -lm -o $@ $^ 

%.o: %.c
	$(CC) -g -MMD -I. -c -o $@ $<

-include $(DEP)

.PHONY: clean

clean:
	rm -f $(OBJ) $(DEP)