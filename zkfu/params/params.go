// Package params holds every compile-time constant of the protocol. The circuit, the native
// reference trainer, the commitment and the server all read from here, so they cannot drift.
package params

const (
	D     = 64   // feature dimension of the frozen extractor (input of the proved head)
	C     = 10   // classes (output of the proved head)
	B     = 32   // batch size of one proved SGD step
	N     = 2048 // padded dataset size per client
	Depth = 11   // Merkle depth, N = 2^Depth
	Frac  = 16   // fractional bits of the fixed-point encoding
	S     = 8    // LR_SHIFT: eta / B = 2^-S
	K     = 1    // local SGD steps per round (one proof = one step)

	// Every private numeric value (feature, weight, bias, new weight, new bias) must lie in
	// [-2^24, 2^24). In-circuit this is checked as  v + 2^24  in  [0, 2^25).
	ValBits   = 25
	ValOffset = 1 << 24

	// Intermediate widths n of the three truncations (bound derivation: DESIGN_NOTES.md §3).
	NFwd = 56 // trunc( sum_d W[c][d]*h[d], Frac )         |x| < 2^54
	NGW  = 70 // trunc( G_W[c][d],          Frac + S )     |x| < 2^68
	NGb  = 46 // trunc( G_b[c],             S )            |x| < 2^44

	// Hash-input packing: 9 range-checked 25-bit values per field element (225 < 253 bits).
	PackPer = 9

	IdxBits    = Depth // bits per batch index
	IdxPerElem = 16    // 16 * 11 = 176 bits per packed public input
	NIdxElems  = B / IdxPerElem

	One = 1 << Frac // fixed-point 1.0 (one-hot target value)
)

// NumPacked returns how many field elements n packed values occupy.
func NumPacked(n int) int { return (n + PackPer - 1) / PackPer }

// InRange reports whether v is a valid private numeric value.
func InRange(v int64) bool { return v >= -ValOffset && v < ValOffset }

func init() {
	if 1<<Depth != N || B%IdxPerElem != 0 || IdxPerElem*IdxBits > 250 || PackPer*ValBits > 250 {
		panic("params: inconsistent constants")
	}
}
