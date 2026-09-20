# chess-engine

A small C++ engine aiming to play chess as best as possible. Currently is able to achieve super-human play. Uses a fail-soft alpha-beta negamax as the core search function, and a modern Efficiently Updatable Neural Network (NNUE), with an old Handcrafted Evaluation (HCE) polyfill.
<br>

## Compiling/running

The NNUE inference uses AVX2 SIMD instructions, which may not be compatible on some systems.
<br><br>
Simply compile with GCC (or MSVC):
<br>
<code>g++ main.cpp -O3 -march=native</code>
<br><br>Download the NNUE network weights (printed in raw weight values as .txt) in the same directory.
<br><br>The engine only supports a few UCI commands needed for communication with other engines, but comes with a better readable CLI output option in the source code.

## Features

Uses traditional fancy magic bitboards for move generation, alongside classic search ordering and pruning techniques to minimize the game tree's size. Includes multi-threading via Lazy-SMP with a lockless transposition table.
<br><br>
Move ordering uses include history tables, continuation histories, countermoves, and categorized by Static Exchange Evaluation (SEE).
<br><br>
The search function prunes the tree using common techniques such as null move pruning, futility pruning, mate distance pruning, and late move reductions/pruning.
<br><br>
The current NNUE architecture is a <code>(768hm -> 512)x2 -> 1x8</code> network: 512 hidden layers, 8 output buckets, 2 accumulators for efficient updates, with horizontal mirroring, activated by SCReLU. Uses quantization and SIMD during updates and inference. Trained on ~459M chess positions with a homemade supervised neural network trainer.
