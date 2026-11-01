# Would a stronger language have caught the 1995 bugs?

The 1995 programs (`src/borland/`, Borland C++ 3.1 compiling C) had nine code bugs, #10–#18; #19 is a figure mismatch, not a code bug. This note asks which of them a stricter language, or modern C++ idioms, would have caught, using the C++23 port (`src/clang/anneq/`) as the worked example.

| Bug | Caught? | How |
|---|---|---|
| [#12](https://github.com/teknoplop/anneq/issues/12) `fill_rand` writes one weight past the array | **Yes** | Run-time bounds checks (Pascal/Ada ranges, Rust, C++ hardened `std::vector`) stop it, and sizing from the container prevents it |
| [#17](https://github.com/teknoplop/anneq/issues/17) swapped `print_weights` arguments, NULL dereference, use-after-free | **Mostly** | References rule out the NULL and RAII (or Rust ownership) rules out the use-after-free; the swap itself needs distinct layer types |
| [#18](https://github.com/teknoplop/anneq/issues/18) `void main`, `argv[0]` trimming, `for(i=i;…)` | **Mostly** | `void main` is a C++ compile error, `i=i` draws a warning, and `std::filesystem` replaces the trimming |
| [#10](https://github.com/teknoplop/anneq/issues/10) recurrent feedback never wired | **By idiom** | The bug was a wrong count passed with a pointer. Passing the whole input container leaves no count to get wrong, but no type checker would object |
| [#13](https://github.com/teknoplop/anneq/issues/13) bias input never set | **By design** | Zero is a legal value in any language; a named bias field that must be set makes the omission visible |
| [#15](https://github.com/teknoplop/anneq/issues/15) random generator reseeded every trial | **Partly** | C++ `<random>` and numpy generators encourage seeding once, but nothing enforces it |
| [#11](https://github.com/teknoplop/anneq/issues/11) RTRL term counted `nodes_1` times | **No** | A maths error in a correctly typed loop, though in effect only a learning-rate scale |
| [#14](https://github.com/teknoplop/anneq/issues/14) retraining loop with no cap | **No** | Logic |
| [#16](https://github.com/teknoplop/anneq/issues/16) training preamble repeats a 31-bit sequence | **No** | A design choice |

**In short:** 3 of the 9 would have been caught outright (#12, #17, #18), and 2 more made hard to write by modern container idioms (#10, #13). The other 4 are algorithm or design errors that no language catches. The bug that mattered most, #10, is in the second group. In C an off-by-count in pointer-plus-length code compiles silently; with containers there is no count to get wrong.

## Side by side

**#10.** The 1995 call is given the number of inputs, and the feedback inputs fall outside it:

```c
ffwd_sig(ann.layer_0, ann.layer_1, ann.nodes_0+1, ann.nodes_1);
```

The C++23 port builds the input vector from named parts, `layer_0` = taps, bias and feedback, so the feedback is part of what is passed. The 1995 behaviour has to be asked for explicitly, by passing `n_in = fb`, the start of the feedback block:

```cpp
for (auto [input, node] : rv::zip(ann.layer_0.feedback, ann.layer_1))
{
    input.output = node.output;
}
ffwd_sig(ann.layer_0.gather(u), ann.layer_1, n_in);   // n_in = n_all when fixed, fb as in 1995
```

**#12.** The 1995 code writes `nodes_0+nodes_1+1` weights into an array of `nodes_0+1`:

```c
fill_rand(ann.layer_1, ann.nodes_0+ann.nodes_1+1, ann.nodes_1);
```

The C++23 port takes the count from the container. The hardened debug build (`make -C src/clang/anneq debug`) aborts at the first out-of-range write if the 1995 count is used:

```cpp
fill_rand(ann.layer_1, ann.layer_0.size());
```

**#17.** The 1995 debug dump passes the layers in the wrong order. The input layer has no weights, so the call dereferences NULL. On real-mode DOS that silently reads low memory; on a modern OS it crashes. In the C++23 port the weights are vectors, not pointers, so an empty layer has an empty weights vector and there is nothing NULL to dereference.

## Takeaway

A stricter language would have caught the memory and housekeeping bugs. It would not by itself have caught the errors that biased the results. #10 needed the container idiom, not the type checker, nothing catches #14 or #16, and #11 turned out to be only a learning-rate scale. Those needed what the 2026 rebuild did: an independent second implementation compared byte for byte, and quick enough runs to look at the results critically.
