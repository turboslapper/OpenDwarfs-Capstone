#include <cstdint>
#include <vector>

/*
==============================================================
 N-Queens (Serial) — Clear, Loop-Only, Bit-Mask Implementation
==============================================================

OVERVIEW
--------
Somehwta similar in concept to OpenDwarfs nqueeens solver. 
We count how many ways to place N queens on an N×N board so that no two
attack each other. We place queens column-by-column from left to right.

BITBOARD BASICS
---------------
We represent rows with a 64-bit integer (a "bitboard"):
- Bit r (0-based; LSB is row 0) represents "row r".
- 1ULL is the 64-bit unsigned literal 1.
  Examples:
    1ULL        -> 0b...0001
    1ULL << 3   -> 0b...1000 (decimal 8)  == "row 3"
- board_mask has the low N bits set to 1:
    N=4  -> 0b0000...1111
    N=8  -> 0b0000...11111111
  Built as: (N >= 64) ? ~0ULL : ((1ULL << N) - 1ULL)
  (Shifting by 64 is undefined; hence the N>=64 guard.)

WHAT ARE masks / left_masks / right_masks / ms?
-----------------------------------------------
Depth i = we are choosing the row for column i.

- masks[i]       : rows already USED by earlier columns (same-row conflicts)
- left_masks[i]  : rows blocked by left-diagonals from earlier queens
- right_masks[i] : rows blocked by right-diagonals from earlier queens
- ms[i]          : the union of everything that makes a row UNAVAILABLE
                   at column i, *plus* the rows we have already TRIED
                   at column i. We add a tried row with: ms[i] |= ns.

Picking the next candidate row at column i:
- Let m  = ms[i].
- Compute ns = (m + 1ULL) & ~m.
  This isolates the lowest ZERO bit of m — i.e., the next available row.
  If (ns & board_mask) == 0, there’s no valid row left in this column.

DIAGONAL SHIFTING
-----------------
If we place a queen at row r in column i (ns == 1ULL << r), then for
the *next* column (i+1):
- That queen blocks row r again (same-row): goes into masks[i+1].
- Left diagonal moves "down-left": (left_masks[i] | ns) << 1.
- Right diagonal moves "down-right": (right_masks[i] | ns) >> 1.

SERIAL SPLIT BY FIRST COLUMN
----------------------------
We fix the queen in column 0 at each possible row r0 (0..N-1) and
solve the rest (columns 1..N-1) serially, summing the results.

SANITY COUNTS (for reference)
-----------------------------
N=8  -> 92
N=10 -> 724
N=12 -> 14200
N=14 -> 365596
==============================================================
*/

// Solve with column 0 fixed at row r0, using the iterative mask stack.
static std::uint64_t count_with_first_row_fixed(int N, int r0)
{
    if (N == 1) return 1;

    // board_mask: low N bits set to 1.
    const std::uint64_t board_mask = (N >= 64) ? ~0ULL : ((1ULL << N) - 1ULL);

    // first: one-hot bit for row r0 (e.g., r0=3 -> 1ULL<<3 == 0b...1000)
    const std::uint64_t first = (1ULL << r0);

    // Per-depth arrays (depth i == column i we’re placing).
    std::vector<std::uint64_t> masks(N + 1, 0);       // rows already used
    std::vector<std::uint64_t> left_masks(N + 1, 0);  // rows blocked by left diagonals
    std::vector<std::uint64_t> right_masks(N + 1, 0); // rows blocked by right diagonals
    std::vector<std::uint64_t> ms(N + 1, 0);          // UNAVAILABLE ∪ TRIED at column i

    // Place queen at (col 0, row r0). Start exploring from column 1.
    masks[1]       = first;
    left_masks[1]  = first << 1; // blocks row r0+1 at next column (if within board)
    right_masks[1] = first >> 1; // blocks row r0-1 at next column (if within board)
    ms[1]          = masks[1] | left_masks[1] | right_masks[1];

    std::uint64_t solutions = 0;
    int i = 1; // current column (1..N-1)

    while (i >= 1) {
        // ns: next available row bit for column i.
        // (ms[i] + 1ULL) finds the lowest zero bit; & ~ms[i] keeps only that bit.
        std::uint64_t m  = ms[i];
        std::uint64_t ns = (m + 1ULL) & ~m;

        if ((ns & board_mask) != 0) {
            // Mark this row as tried so next loop tries the next one.
            ms[i] |= ns;

            if (i == N - 1) {
                // Last column placed -> one complete solution.
                ++solutions;
            } else {
                // Propagate constraints to the next column (i+1).
                masks[i + 1]       = masks[i] | ns;
                left_masks[i + 1]  = (left_masks[i] | ns) << 1;
                right_masks[i + 1] = (right_masks[i] | ns) >> 1;

                // ms[i+1]: all rows blocked at column i+1 (no "tried" rows there yet).
                ms[i + 1]          = masks[i + 1] | left_masks[i + 1] | right_masks[i + 1];

                ++i; // go deeper
            }
        } else {
            // No more candidates in this column -> backtrack.
            --i;
        }
    }
    return solutions;
}

// This is the serial function your main() calls.
std::uint64_t count_nqueens_parallel(int n)
{
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (n > 63)  return 0; // stay within safe 64-bit shifting

    std::uint64_t total = 0;

    // NQ outler loop
    for (int r0 = 0; r0 < n; ++r0) {
        total += count_with_first_row_fixed(n, r0);
    }
    return total;
}
