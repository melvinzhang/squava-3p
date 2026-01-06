// +build amd64

#include "textflag.h"

// Lane order: 0: Horizontal (s=1), 1: Vertical (s=8), 2: Diagonal (s=9), 3: Anti-diagonal (s=7)

DATA ·shifts1+0(SB)/8, $1
DATA ·shifts1+8(SB)/8, $8
DATA ·shifts1+16(SB)/8, $9
DATA ·shifts1+24(SB)/8, $7
GLOBL ·shifts1(SB), RODATA, $32

DATA ·shifts2+0(SB)/8, $2
DATA ·shifts2+8(SB)/8, $16
DATA ·shifts2+16(SB)/8, $18
DATA ·shifts2+24(SB)/8, $14
GLOBL ·shifts2(SB), RODATA, $32

DATA ·shifts3+0(SB)/8, $3
DATA ·shifts3+8(SB)/8, $24
DATA ·shifts3+16(SB)/8, $27
DATA ·shifts3+24(SB)/8, $21
GLOBL ·shifts3(SB), RODATA, $32

// MaskR1: MaskNotH, ALL, MaskNotH, MaskNotA
DATA ·maskR1+0(SB)/8, $0x7F7F7F7F7F7F7F7F
DATA ·maskR1+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskR1+16(SB)/8, $0x7F7F7F7F7F7F7F7F
DATA ·maskR1+24(SB)/8, $0xFEFEFEFEFEFEFEFE
GLOBL ·maskR1(SB), RODATA, $32

// MaskL1: MaskNotA, ALL, MaskNotA, MaskNotH
DATA ·maskL1+0(SB)/8, $0xFEFEFEFEFEFEFEFE
DATA ·maskL1+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskL1+16(SB)/8, $0xFEFEFEFEFEFEFEFE
DATA ·maskL1+24(SB)/8, $0x7F7F7F7F7F7F7F7F
GLOBL ·maskL1(SB), RODATA, $32

// MaskR2: MaskNotGH, ALL, MaskNotGH, MaskNotAB
DATA ·maskR2+0(SB)/8, $0x3F3F3F3F3F3F3F3F
DATA ·maskR2+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskR2+16(SB)/8, $0x3F3F3F3F3F3F3F3F
DATA ·maskR2+24(SB)/8, $0xFCFCFCFCFCFCFCFC
GLOBL ·maskR2(SB), RODATA, $32

// MaskL2: MaskNotAB, ALL, MaskNotAB, MaskNotGH
DATA ·maskL2+0(SB)/8, $0xFCFCFCFCFCFCFCFC
DATA ·maskL2+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskL2+16(SB)/8, $0xFCFCFCFCFCFCFCFC
DATA ·maskL2+24(SB)/8, $0x3F3F3F3F3F3F3F3F
GLOBL ·maskL2(SB), RODATA, $32

// MaskR3: MaskNotFGH, ALL, MaskNotFGH, MaskNotABC
DATA ·maskR3+0(SB)/8, $0x1F1F1F1F1F1F1F1F
DATA ·maskR3+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskR3+16(SB)/8, $0x1F1F1F1F1F1F1F1F
DATA ·maskR3+24(SB)/8, $0xF8F8F8F8F8F8F8F8
GLOBL ·maskR3(SB), RODATA, $32

// MaskL3: MaskNotABC, ALL, MaskNotABC, MaskNotFGH
DATA ·maskL3+0(SB)/8, $0xF8F8F8F8F8F8F8F8
DATA ·maskL3+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·maskL3+16(SB)/8, $0xF8F8F8F8F8F8F8F8
DATA ·maskL3+24(SB)/8, $0x1F1F1F1F1F1F1F1F
GLOBL ·maskL3(SB), RODATA, $32

// func getWinsAndLossesAVX2(b, e uint64) (w, l uint64)
TEXT ·getWinsAndLossesAVX2(SB), NOSPLIT, $0-32
    MOVQ b+0(FP), AX
    VMOVQ AX, X0
    VPBROADCASTQ X0, Y0      // Y0 = [b, b, b, b]
    MOVQ e+8(FP), BX
    VMOVQ BX, X1
    VPBROADCASTQ X1, Y1      // Y1 = [e, e, e, e]
    
    // Shifts 1
    VMOVDQU ·shifts1(SB), Y2
    VPSRLVQ Y2, Y0, Y3       // Y3 = b >> s
    VPAND ·maskR1(SB), Y3, Y3 // Y3 = r1
    VPSLLVQ Y2, Y0, Y4       // Y4 = b << s
    VPAND ·maskL1(SB), Y4, Y4 // Y4 = l1
    
    // Shifts 2
    VMOVDQU ·shifts2(SB), Y2
    VPSRLVQ Y2, Y0, Y5       // Y5 = b >> 2s
    VPAND ·maskR2(SB), Y5, Y5 // Y5 = r2
    VPSLLVQ Y2, Y0, Y6       // Y6 = b << 2s
    VPAND ·maskL2(SB), Y6, Y6 // Y6 = l2
    
    // Shifts 3
    VMOVDQU ·shifts3(SB), Y2
    VPSRLVQ Y2, Y0, Y7       // Y7 = b >> 3s
    VPAND ·maskR3(SB), Y7, Y7 // Y7 = r3
    VPSLLVQ Y2, Y0, Y8       // Y8 = b << 3s
    VPAND ·maskL3(SB), Y8, Y8 // Y8 = l3
    
    // Calculate L lanes: e & (r1&r2 | r1&l1 | l1&l2)
    VPAND Y3, Y5, Y9         // r1 & r2
    VPAND Y3, Y4, Y10        // r1 & l1
    VPOR Y9, Y10, Y9         // r1&r2 | r1&l1
    VPAND Y4, Y6, Y10        // l1 & l2
    VPOR Y10, Y9, Y9         // r1&r2 | r1&l1 | l1&l2
    VPAND Y1, Y9, Y9         // Y9 = L lanes
    
    // Calculate W lanes: e & (r1&r2&(r3|l1) | l1&l2&(r1|l3))
    VPOR Y7, Y4, Y10         // r3 | l1
    VPAND Y3, Y5, Y11        // r1 & r2
    VPAND Y11, Y10, Y10      // r1&r2&(r3|l1)
    
    VPOR Y3, Y8, Y11         // r1 | l3
    VPAND Y4, Y6, Y12        // l1 & l2
    VPAND Y12, Y11, Y11      // l1&l2&(r1|l3)
    
    VPOR Y10, Y11, Y10       // r1&r2&(r3|l1) | l1&l2&(r1|l3)
    VPAND Y1, Y10, Y10       // Y10 = W lanes
    
    // Horizontal OR for W
    VEXTRACTI128 $1, Y10, X11 // X11 = upper 128 bits of Y10
    VPOR X11, X10, X10        // X10 = [W0|W2, W1|W3]
    VPSHUFD $0x4E, X10, X11   // swap 64-bit halves
    VPOR X11, X10, X10        // X10 = [W0|W1|W2|W3, ...]
    MOVQ X10, AX
    MOVQ AX, w+16(FP)
    
    // Horizontal OR for L
    VEXTRACTI128 $1, Y9, X11
    VPOR X11, X9, X9
    VPSHUFD $0x4E, X9, X11
    VPOR X11, X9, X9
    MOVQ X9, AX
    MOVQ AX, l+24(FP)
    
    VZEROUPPER
    RET

// func pdep(src, mask uint64) uint64
TEXT ·pdep(SB), $0-24
    MOVQ src+0(FP), AX
    MOVQ mask+8(FP), CX
    PDEPQ CX, AX, AX
    MOVQ AX, ret+16(FP)
    RET

DATA ·asmIndices+0(SB)/4, $0
DATA ·asmIndices+4(SB)/4, $1
DATA ·asmIndices+8(SB)/4, $2
DATA ·asmIndices+12(SB)/4, $3
DATA ·asmIndices+16(SB)/4, $4
DATA ·asmIndices+20(SB)/4, $5
DATA ·asmIndices+24(SB)/4, $6
DATA ·asmIndices+28(SB)/4, $7
GLOBL ·asmIndices(SB), RODATA, $32

DATA ·asmNegInf(SB)/4, $0xff800000
GLOBL ·asmNegInf(SB), RODATA, $4

// func selectBestEdgeAVX2(qs []float32, us []float32, coeff float32) int
TEXT ·selectBestEdgeAVX2(SB), NOSPLIT, $0-64
    MOVQ qs_base+0(FP), SI
    MOVQ qs_len+8(FP), CX      // CX = length
    MOVQ us_base+24(FP), DI
    VMOVSS coeff+48(FP), X0
    VBROADCASTSS X0, Y0       // Y0 = [coeff...]

    VMOVDQU ·asmIndices(SB), Y3   // Y3 = current indices [0..7]
    VMOVSS ·asmNegInf(SB), X1
    VBROADCASTSS X1, Y1       // Y1 = best scores (-inf)
    VPXOR Y2, Y2, Y2          // Y2 = best indices (0)

    MOVQ $8, AX
    VMOVQ AX, X4
    VPBROADCASTD X4, Y4       // Y4 = [8...]

    MOVQ $0, DX               // loop counter
loop:
    MOVQ CX, BX
    SUBQ DX, BX
    CMPQ BX, $8
    JL reduce

    VMOVUPS (SI)(DX*4), Y5    // load 8 Qs
    VMOVUPS (DI)(DX*4), Y6    // load 8 Us
    
    VFMADD213PS Y5, Y0, Y6    // Y6 = Y0*Y6 + Y5 = coeff*U + Q
    
    VCMPPS $14, Y1, Y6, Y8    // Y8 = Y6 > Y1
    
    VBLENDVPS Y8, Y6, Y1, Y1  // update best scores
    VBLENDVPS Y8, Y3, Y2, Y2  // update best indices
    
    VPADDD Y4, Y3, Y3         // current indices += 8
    ADDQ $8, DX
    JMP loop

reduce:
    // Reduce Y1 (scores) and Y2 (indices) to scalar X1/X2 lane 0
    VEXTRACTI128 $1, Y1, X5
    VEXTRACTI128 $1, Y2, X6
    VCMPPS $14, X1, X5, X8
    VBLENDVPS X8, X5, X1, X1
    VBLENDVPS X8, X6, X2, X2

    VPSHUFD $0x4E, X1, X5
    VPSHUFD $0x4E, X2, X6
    VCMPPS $14, X1, X5, X8
    VBLENDVPS X8, X5, X1, X1
    VBLENDVPS X8, X6, X2, X2

    VPSHUFD $0xB1, X1, X5
    VPSHUFD $0xB1, X2, X6
    VCMPPS $14, X1, X5, X8
    VBLENDVPS X8, X5, X1, X1
    VBLENDVPS X8, X6, X2, X2

    VMOVD X2, R9              // R9 = current best index

remainder:
    CMPQ DX, CX
    JGE done_final
    
    VMOVSS (SI)(DX*4), X5     // Q
    VMOVSS (DI)(DX*4), X6     // U
    VFMADD213SS X5, X0, X6    // X6 = X0 * X6 + X5 = coeff * U + Q
    
    VCOMISS X1, X6            // compare current score (X6) with best (X1)
    JBE next_rem
    VMOVUPS X6, X1
    MOVQ DX, R9               // update best index
next_rem:
    ADDQ $1, DX
    JMP remainder

done_final:
    MOVQ R9, AX
    MOVQ AX, ret+56(FP)
    VZEROUPPER
    RET
