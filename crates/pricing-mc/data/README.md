# Joe--Kuo Sobol direction data

`joe-kuo-6.21201-u32be.bin` embeds the complete 21,201-dimensional,
32-bit direction matrix for the Joe--Kuo `new-joe-kuo-6.21201` set.

The file format is deterministic:

- bytes 0--7: ASCII magic `JK621201`;
- bytes 8--11: big-endian format version `1`;
- bytes 12--15: big-endian dimension count `21201`;
- bytes 16--19: big-endian bit count `32`;
- remaining bytes: row-major, big-endian `u32` direction words, with Sobol
  dimension as the outer axis and direction bit as the inner axis.

The asset was generated from SciPy's `_sobol_direction_numbers.npz`, which
attributes the data to Frances Y. Kuo's UNSW Sobol sequence resource. The
upstream recommended file is `new-joe-kuo-6.21201`, updated 5 January 2010.
See <https://web.maths.unsw.edu.au/~fkuo/sobol/>.

The SHA-256 digest of the complete embedded file is
`189f65c4e4fcf7455efb7618f3dafbbbaf70303fc35ecd380fa28a67ab896900`.

SciPy is distributed under the BSD 3-Clause License. Its license and third
party notices are available at <https://github.com/scipy/scipy/blob/main/LICENSE.txt>.
