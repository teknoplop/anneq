// bp_engine.v - one BPBC 2-6-6-1 batch-backprop trainer (one trial). Bit-exact with gateware/ref/bpfix.c.
// Single pipelined 18x18 MAC (M1, also used in chunks for the epoch update), M2 = h*h, M3 = der*x.
// Pipeline: issue -> s1 (memory data, operands) -> s2 (product) -> s3 (accumulate / gradient RMW)
//           -> s4 (tanh LUT read | der*x) -> s5 (write activation | delta).
`default_nettype none
module bp_engine (
    input  wire        clk,
    input  wire        start,
    input  wire [7:0]  nsamp,
    input  wire [15:0] epochs,
    input  wire [17:0] k1,
    input  wire [17:0] k2,
    output reg         busy,
    input  wire        h_we,
    input  wire        h_mem,      // 0 = weights, 1 = training data
    input  wire [7:0]  h_addr,
    input  wire [31:0] h_wdata,
    output wire [31:0] h_rdata
);
    // ---------------- memories
    reg [31:0] wmem [0:255];        // 0..66 weights, 128..194 momentum deltas
    reg [35:0] gmem [0:127];
        (* ram_style = "distributed" *) reg [16:0] xmem [0:63];
    reg [13:0] lut  [0:2047];
    initial $readmemh("tanh_lut.hex", lut);

    reg  [7:0] w_ra, w_wa;  reg [6:0] g_ra;  reg [5:0] x_ra;   // combinational read addresses
    reg [31:0] wq;  reg [35:0] gq;  reg [16:0] xq;
    reg        w_we, g_we;  reg [6:0] g_wa;
    reg [31:0] w_wd;  reg [35:0] g_wd;
    always @(posedge clk) begin
        wq <= wmem[w_ra]; gq <= gmem[g_ra]; xq <= xmem[x_ra];
        if (w_we) wmem[w_wa] <= w_wd;
        if (g_we) gmem[g_wa] <= g_wd;
        if (h_we && h_mem && !busy) xmem[h_addr[5:0]] <= h_wdata[16:0];
    end
    assign h_rdata = wq;

    // ---------------- registers: act 0..2 a0, 3..9 a1, 10..16 a2, 17 y ; dd 0..5 d1, 6..11 d2, 12 d3
    reg signed [17:0] act [0:17];
    reg signed [17:0] dd  [0:12];
    reg signed [17:0] dr  [0:17];   // der(act) = 1 - act^2, written with each activation
    reg D;   // desired bit of current pattern

    // ---------------- control
    localparam S_IDLE = 0, S_CLR = 1, S_SS0 = 2, S_SS1 = 3, S_SS2 = 4, S_RUN = 5, S_DRAIN = 6, S_UPD = 7;
    localparam P_F1 = 0, P_F2 = 1, P_F3 = 2, P_B3 = 3, P_B2 = 4, P_B1 = 5, P_G3 = 6, P_G2 = 7, P_G1 = 8;
    reg [2:0] st;
    reg [3:0] ph;
    reg [2:0] n, j;
    reg [7:0] samp;
    reg [15:0] ep;
    reg [6:0] uidx;
    reg [3:0] ustep;

    // phase geometry
    reg [2:0] NN, JJ;
    always @* begin
        case (ph)
            P_F1: begin NN = 6; JJ = 3; end
            P_F2: begin NN = 6; JJ = 7; end
            P_F3: begin NN = 1; JJ = 7; end
            P_B3: begin NN = 1; JJ = 1; end
            P_B2: begin NN = 6; JJ = 1; end
            P_B1: begin NN = 6; JJ = 6; end
            P_G3: begin NN = 1; JJ = 7; end
            P_G2: begin NN = 6; JJ = 7; end
            default: begin NN = 6; JJ = 3; end
        endcase
    end
    wire lastj = (j == JJ - 1), lastn = (n == NN - 1);
    wire issuing = (st == S_RUN);
    wire [6:0] n3 = {n, 1'b0} + n, n7 = {n, 3'b0} - n, j7 = {j, 3'b0} - j;

    // issue-stage operands
    reg  [6:0] widx;  reg usew;  reg signed [17:0] ia, ib, ih;  reg [1:0] kind;  reg [4:0] dest;
    wire signed [17:0] Dv = D ? 18'sd16384 : -18'sd16384;
    always @* begin
        widx = 0; usew = 1; ia = 0; ib = 0; ih = 0; kind = 0; dest = 0;
        case (ph)
            P_F1: begin widx = n3 + j;      ib = act[j];      dest = 3 + n;  end
            P_F2: begin widx = 18 + n7 + j; ib = act[3 + j];  dest = 10 + n; end
            P_F3: begin widx = 60 + j;       ib = act[10 + j]; dest = 17;     end
            P_B3: begin usew = 0; ia = Dv - act[17]; ib = 18'sd16384; ih = dr[17]; kind = 1; dest = 12; end
            P_B2: begin widx = 60 + n;       ib = dd[12];      ih = dr[10 + n]; kind = 1; dest = 6 + n; end
            P_B1: begin widx = 18 + j7 + n; ib = dd[6 + j];   ih = dr[3 + n];  kind = 1; dest = n; end
            P_G3: begin widx = 60 + j;       usew = 0; ia = dd[12];    ib = act[10 + j]; kind = 2; end
            P_G2: begin widx = 18 + n7 + j; usew = 0; ia = dd[6 + n]; ib = act[3 + j];  kind = 2; end
            default: begin widx = n3 + j;   usew = 0; ia = dd[n];     ib = act[j];      kind = 2; end
        endcase
    end

    // ---------------- pipeline registers
    reg v1, v2, v3f, v3b, v4f, v4b, v5f;  reg [4:0] dst5;
    reg [1:0] k1r, k2r;  reg f1, f2, l1, l2;  reg [4:0] dst1, dst2, dst3, dst4;  reg [6:0] gi1, gi2;
    reg usew1;  reg signed [17:0] ia1, ib1, ih1;  reg signed [35:0] g2;
    reg signed [35:0] p;  reg signed [17:0] ih2;
    reg signed [39:0] acc;
    reg [10:0] lidx;  reg sgn3, sgn4;  reg [13:0] lq;
    reg signed [17:0] der3, xs3;  reg signed [35:0] m3;

    // update-phase multiplier operands
    reg signed [17:0] ua, ub;
    wire signed [17:0] opa = (st == S_UPD) ? ua : (usew1 ? $signed(wq[31:14]) : ia1);
    wire signed [17:0] opb = (st == S_UPD) ? ub : ib1;

    // s3 combinational
    wire signed [39:0] sum = f2 ? {{4{p[35]}}, p} : acc + {{4{p[35]}}, p};
    wire signed [25:0] ssh = sum[39:14];
    wire        [25:0] mag = ssh[25] ? -ssh : ssh;
    wire signed [17:0] ssat = (ssh > 26'sd131071) ? 18'sd131071 : (ssh < -26'sd131072) ? -18'sd131072 : ssh[17:0];
    wire signed [21:0] m3s = m3[35:14];
    wire signed [17:0] m3sat = (m3s > 22'sd131071) ? 18'sd131071 : (m3s < -22'sd131072) ? -18'sd131072 : m3s[17:0];

    wire pipe_empty = !(v1 | v2 | v3f | v3b | v4f | v4b | v5f);

    // update registers
    reg [31:0] uw;  reg signed [31:0] ud;  reg signed [39:0] ug;  reg signed [63:0] ut;

    integer i;
    initial begin busy = 0; st = S_IDLE; v1 = 0; v2 = 0; v3f = 0; v3b = 0; v4f = 0; v4b = 0; v5f = 0; end

    // memory address / write muxing (combinational)
    always @* begin
        w_ra = {1'b0, widx}; g_ra = widx; x_ra = samp[5:0];
        w_we = 0; g_we = 0; w_wa = 0; g_wa = 0; w_wd = 0; g_wd = 0;
        if (!busy) begin
            w_ra = h_addr;
            w_we = h_we && !h_mem; w_wa = h_addr; w_wd = h_wdata;
        end
        if (st == S_SS1) x_ra = samp[5:0] + 1;
        if (st == S_UPD) begin w_ra = (ustep == 1) ? {1'b1, uidx} : {1'b0, uidx}; g_ra = uidx; end
        if (st == S_CLR) begin g_we = 1; g_wa = uidx; w_we = 1; w_wa = {1'b1, uidx}; end
        if (k2r == 2 && v2) begin g_we = 1; g_wa = gi2; g_wd = g2 + p; end
        if (st == S_UPD && ustep == 8) begin
            w_we = 1; w_wa = {1'b0, uidx}; w_wd = uw + ut[47:16];
            g_we = 1; g_wa = uidx; g_wd = 0;
        end
        if (st == S_UPD && ustep == 9) begin w_we = 1; w_wa = {1'b1, uidx}; w_wd = ud; end
    end

    always @(posedge clk) begin
        // ---------- datapath pipeline
        v1 <= issuing; k1r <= kind; f1 <= (j == 0); l1 <= lastj; dst1 <= dest; gi1 <= widx;
        usew1 <= usew; ia1 <= ia; ib1 <= ib; ih1 <= ih;
        // s2
        v2 <= v1; k2r <= k1r; f2 <= f1; l2 <= l1; dst2 <= dst1; gi2 <= gi1; g2 <= gq;
        p <= opa * opb;
        ih2 <= ih1;
        // s3
        v3f <= 0; v3b <= 0;
        if (v2 && k2r != 2) begin
            acc <= sum;
            if (l2) begin
                dst3 <= dst2;
                if (k2r == 0) begin v3f <= 1; sgn3 <= ssh[25]; lidx <= (mag[25:5] > 2047) ? 11'd2047 : mag[15:5]; end
                else begin v3b <= 1; der3 <= ih2; xs3 <= ssat; end
            end
        end
        // s4
        v4f <= v3f; v4b <= v3b; dst4 <= dst3; sgn4 <= sgn3;
        lq <= lut[lidx];
        m3 <= v3b ? der3 * xs3 : $signed({4'b0, lq}) * $signed({4'b0, lq});
        v5f <= v4f; dst5 <= dst4;
        // s5
        if (v4f) act[dst4] <= sgn4 ? -$signed({4'b0, lq}) : $signed({4'b0, lq});
        if (v4b) dd[dst4]  <= m3sat;
        if (v5f) dr[dst5] <= 18'sd16384 - m3[31:14];

        // ---------- control
        case (st)
            S_IDLE: if (start) begin busy <= 1; st <= S_CLR; uidx <= 0; ep <= 0; end
            S_CLR: begin uidx <= uidx + 1; if (uidx == 127) begin st <= S_SS0; samp <= 0; end end
            S_SS0: st <= S_SS1;
            S_SS1: begin act[1] <= $signed(xq[15:0]); st <= S_SS2; end
            S_SS2: begin
                act[0] <= $signed(xq[15:0]); D <= xq[16];
                act[2] <= -18'sd16384; act[9] <= -18'sd16384; act[16] <= -18'sd16384;
                st <= S_RUN; ph <= P_F1; n <= 0; j <= 0;
            end
            S_RUN: begin
                if (!lastj) j <= j + 1;
                else begin
                    j <= 0;
                    if (!lastn) n <= n + 1;
                    else begin
                        n <= 0;
                        // gradient sub-phases have no dependency between them: no drain
                        if (ph == P_G3 || ph == P_G2) ph <= ph + 1;
                        else st <= S_DRAIN;
                    end
                end
            end
            S_DRAIN: if (pipe_empty) begin
                if (ph == P_G1) begin
                    if (samp == nsamp - 1) begin st <= S_UPD; uidx <= 0; ustep <= 0; end
                    else begin samp <= samp + 1; st <= S_SS0; end
                end else begin
                    ph <= ph + 1; st <= S_RUN;
                end
            end
            S_UPD: begin
                ustep <= ustep + 1;
                case (ustep)
                    0: ;                                                     // addresses out
                    1: begin uw <= wq; ug <= {{4{gq[35]}}, gq}; end
                    2: begin ud <= wq; ut <= 0; end
                    3: ut <= ut + p;
                    4: ut <= ut + (p <<< 17);
                    5: ut <= ut + (p <<< 34);
                    6: ut <= ut + p;
                    7: ut <= ut + (p <<< 17);
                    8: ud <= ut[47:16];
                    9: begin
                        ut <= 0; ustep <= 0;
                        if (uidx == 66) begin
                            uidx <= 0;
                            if (ep == epochs - 1) begin st <= S_IDLE; busy <= 0; end
                            else begin ep <= ep + 1; samp <= 0; st <= S_SS0; end
                        end else uidx <= uidx + 1;
                    end
                endcase
            end
            default: st <= S_IDLE;
        endcase
    end

    // multiplier operand sequence for the update (product appears one cycle after the operands)
    always @* begin
        ua = 0; ub = 0;
        case (ustep)
            2: begin ua = {1'b0, ug[16:0]};  ub = k1; end
            3: begin ua = {1'b0, ug[33:17]}; ub = k1; end
            4: begin ua = {{12{ug[39]}}, ug[39:34]}; ub = k1; end
            5: begin ua = {1'b0, ud[16:0]};  ub = k2; end
            6: begin ua = {{3{ud[31]}}, ud[31:17]}; ub = k2; end
            default: ;
        endcase
    end
endmodule
`default_nettype wire
