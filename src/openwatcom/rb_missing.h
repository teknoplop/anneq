/* RB.C as printed in the report has no declaration of its loop counter bit_no
   (its first listing page, with the file header, is missing from the bound copy
   and the declaration list jumps straight from delay_count to the code).
   Supplied here as a file-scope int, the only change needed to compile it. */
int bit_no;
