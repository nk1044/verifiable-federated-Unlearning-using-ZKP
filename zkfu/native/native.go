// Package native is the reference fixed-point trainer. It is bit-identical to the circuit
// (tests/step_equiv_test.go checks 1000 random steps). This is the ONLY place, together with the
// circuit, where weight updates are computed.
package native

import (
	"errors"
	"fmt"
	"math/big"

	"zkfu/params"
)

var ErrOverflow = errors.New("fixed-point: value outside the truncation width")

var one = big.NewInt(1)

// Trunc(x, t) = floor(x / 2^t) for signed x with |x| < 2^(n-1), computed exactly as the circuit
// does: x' = x + 2^(n-1) = q*2^t + r with 0 <= r < 2^t, 0 <= q < 2^(n-t); output q - 2^(n-1-t).
func Trunc(x *big.Int, t, n uint) (*big.Int, error) {
	xp := new(big.Int).Add(x, new(big.Int).Lsh(one, n-1))
	if xp.Sign() < 0 || xp.BitLen() > int(n) {
		return nil, ErrOverflow
	}
	q := new(big.Int).Rsh(xp, t)
	return q.Sub(q, new(big.Int).Lsh(one, n-1-t)), nil
}

// Step performs one proved fixed-point SGD step on the linear head with MSE loss.
//
//	z_c  = trunc(sum_d W[c][d]*h[d], Frac) + b[c]
//	e_c  = z_c - y_c                      (y_c = 2^Frac for the true class, else 0)
//	G_W  = sum_j m_j e_j[c] h_j[d] ;  G_b = sum_j m_j e_j[c]
//	W'   = W - trunc(G_W, Frac+S) ;   b'  = b - trunc(G_b, S)
//
// It rejects inputs outside the range the circuit accepts, and outputs outside it (the circuit
// range-checks W', b' too), so an honest client never produces an unprovable update silently.
func Step(W [][]int64, b []int64, H [][]int64, Y []int64, M []uint8) ([][]int64, []int64, error) {
	if len(W) != params.C || len(b) != params.C || len(H) != params.B || len(Y) != params.B || len(M) != params.B {
		return nil, nil, fmt.Errorf("step: wrong shapes")
	}
	for c := 0; c < params.C; c++ {
		if len(W[c]) != params.D || !params.InRange(b[c]) {
			return nil, nil, fmt.Errorf("step: bad head row %d", c)
		}
		for d := 0; d < params.D; d++ {
			if !params.InRange(W[c][d]) {
				return nil, nil, fmt.Errorf("step: W[%d][%d] out of range", c, d)
			}
		}
	}
	for j := 0; j < params.B; j++ {
		if len(H[j]) != params.D || Y[j] < 0 || Y[j] >= params.C || M[j] > 1 {
			return nil, nil, fmt.Errorf("step: bad sample %d", j)
		}
		for d := 0; d < params.D; d++ {
			if !params.InRange(H[j][d]) {
				return nil, nil, fmt.Errorf("step: feature %d,%d out of range", j, d)
			}
		}
	}

	// forward + masked error  me[j][c] = m_j * e_j[c]
	me := make([][]*big.Int, params.B)
	tmp := new(big.Int)
	for j := 0; j < params.B; j++ {
		me[j] = make([]*big.Int, params.C)
		for c := 0; c < params.C; c++ {
			acc := new(big.Int)
			for d := 0; d < params.D; d++ {
				acc.Add(acc, tmp.Mul(big.NewInt(W[c][d]), big.NewInt(H[j][d])))
			}
			z, err := Trunc(acc, params.Frac, params.NFwd)
			if err != nil {
				return nil, nil, err
			}
			z.Add(z, big.NewInt(b[c]))
			if Y[j] == int64(c) {
				z.Sub(z, big.NewInt(params.One))
			}
			if M[j] == 0 {
				z.SetInt64(0)
			}
			me[j][c] = z
		}
	}

	W2 := make([][]int64, params.C)
	b2 := make([]int64, params.C)
	for c := 0; c < params.C; c++ {
		W2[c] = make([]int64, params.D)
		for d := 0; d < params.D; d++ {
			g := new(big.Int)
			for j := 0; j < params.B; j++ {
				g.Add(g, tmp.Mul(me[j][c], big.NewInt(H[j][d])))
			}
			t, err := Trunc(g, params.Frac+params.S, params.NGW)
			if err != nil {
				return nil, nil, err
			}
			v := new(big.Int).Sub(big.NewInt(W[c][d]), t)
			if !v.IsInt64() || !params.InRange(v.Int64()) {
				return nil, nil, fmt.Errorf("step: W'[%d][%d] out of range", c, d)
			}
			W2[c][d] = v.Int64()
		}
		gb := new(big.Int)
		for j := 0; j < params.B; j++ {
			gb.Add(gb, me[j][c])
		}
		t, err := Trunc(gb, params.S, params.NGb)
		if err != nil {
			return nil, nil, err
		}
		v := new(big.Int).Sub(big.NewInt(b[c]), t)
		if !v.IsInt64() || !params.InRange(v.Int64()) {
			return nil, nil, fmt.Errorf("step: b'[%d] out of range", c)
		}
		b2[c] = v.Int64()
	}
	return W2, b2, nil
}
