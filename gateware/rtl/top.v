// top.v - Icepi Zero: NENG bp_engine instances behind a UART command interface (8N1, BAUD).
// Commands (little endian):
//   'W' e m aL aH d0 d1 d2 d3 : write. m=0 weight[a] of engine e, m=1 training word[a] of engine e,
//                               m=3 global register a (0 nsamp, 1 epochs, 2 K1, 3 K2, 4 engine enable mask)
//   'R' e aL aH               : reply 4 bytes = weight[a] of engine e
//   'G'                       : start enabled engines; when all are done reply 4 bytes = clock cycles taken
`default_nettype none
module top #(parameter NENG = 8, parameter CLK = 50000000, parameter BAUD = 2000000) (
    input  wire clk,
    input  wire usb_rx,
    output wire usb_tx,
    output wire [4:0] led
);
    localparam DIV = CLK / BAUD;

    // ---------------- UART receive
    reg [1:0] rxs = 2'b11;
    always @(posedge clk) rxs <= {rxs[0], usb_rx};
    reg [7:0] rdiv = 0; reg [3:0] rbit = 0; reg [7:0] rsh = 0; reg rbusy = 0; reg rvalid = 0; reg [7:0] rbyte = 0;
    always @(posedge clk) begin
        rvalid <= 0;
        if (!rbusy) begin
            if (!rxs[1]) begin rbusy <= 1; rdiv <= DIV / 2; rbit <= 0; end
        end else if (rdiv == DIV - 1) begin
            rdiv <= 0; rbit <= rbit + 1;
            if (rbit == 0) begin if (rxs[1]) rbusy <= 0; end           // false start
            else if (rbit <= 8) rsh <= {rxs[1], rsh[7:1]};
            else begin rbusy <= 0; rvalid <= 1; rbyte <= rsh; end      // stop bit
        end else rdiv <= rdiv + 1;
    end

    // ---------------- UART transmit
    reg [7:0] tdiv = 0; reg [3:0] tbit = 0; reg [9:0] tsh = 10'h3ff; reg tbusy = 0;
    reg tstart = 0; reg [7:0] tbyte = 0;
    // transmit FIFO: replies are queued so the parser never stalls while bytes go out
    reg [7:0] fifo [0:63]; reg [5:0] fw = 0, fr = 0; reg push = 0; reg [7:0] pbyte = 0;
    always @(posedge clk) begin
        tstart <= 0;
        if (push) begin fifo[fw] <= pbyte; fw <= fw + 1; end
        if (!tbusy && !tstart && fr != fw) begin tbyte <= fifo[fr]; fr <= fr + 1; tstart <= 1; end
    end
    assign usb_tx = tsh[0];
    always @(posedge clk) begin
        if (!tbusy) begin
            if (tstart) begin tbusy <= 1; tsh <= {1'b1, tbyte, 1'b0}; tdiv <= 0; tbit <= 0; end
        end else if (tdiv == DIV - 1) begin
            tdiv <= 0; tsh <= {1'b1, tsh[9:1]}; tbit <= tbit + 1;
            if (tbit == 9) tbusy <= 0;
        end else tdiv <= tdiv + 1;
    end

    // ---------------- engines
    reg [7:0] nsamp = 40; reg [15:0] epochs = 1000; reg [17:0] k1 = 3277, k2 = 32768; reg [NENG-1:0] mask = 0;
    reg start = 0; reg h_we = 0; reg h_mem = 0; reg [7:0] h_addr = 0; reg [31:0] h_wdata = 0; reg [7:0] h_eng = 0;
    wire [NENG-1:0] busy;
    wire [31:0] rd [0:NENG-1];
    genvar g;
    generate for (g = 0; g < NENG; g = g + 1) begin : eng
        bp_engine u (.clk(clk), .start(start & mask[g]), .nsamp(nsamp), .epochs(epochs), .k1(k1), .k2(k2),
                     .busy(busy[g]), .h_we(h_we && h_eng == g), .h_mem(h_mem), .h_addr(h_addr),
                     .h_wdata(h_wdata), .h_rdata(rd[g]));
    end endgenerate

    // ---------------- command parser
    reg [3:0] ps = 0; reg [7:0] cmd = 0; reg [7:0] b [0:7]; reg [3:0] cnt = 0;
    reg [31:0] cycles = 0; reg [31:0] txw = 0; reg [2:0] txn = 0; reg [1:0] wt = 0;
    localparam P_CMD = 0, P_ARG = 1, P_EXEC = 2, P_RUN = 3, P_TX = 4, P_RD = 5;
    always @(posedge clk) begin
        h_we <= 0; start <= 0; push <= 0;
        case (ps)
            P_CMD: if (rvalid) begin
                cmd <= rbyte; cnt <= 0;
                if (rbyte == "W" || rbyte == "R") ps <= P_ARG;
                else if (rbyte == "G") begin start <= 1; cycles <= 0; wt <= 0; ps <= P_RUN; end
            end
            P_ARG: if (rvalid) begin
                b[cnt] <= rbyte; cnt <= cnt + 1;
                if ((cmd == "W" && cnt == 7) || (cmd == "R" && cnt == 2)) ps <= P_EXEC;
            end
            P_EXEC: begin
                h_eng <= b[0]; h_addr <= (cmd == "R") ? b[1] : b[2]; h_wdata <= {b[7], b[6], b[5], b[4]}; h_mem <= b[1][0];
                if (cmd == "W") begin
                    if (b[1] == 3) case (b[2])
                        0: nsamp <= b[4];
                        1: epochs <= {b[5], b[4]};
                        2: k1 <= {b[6][1:0], b[5], b[4]};
                        3: k2 <= {b[6][1:0], b[5], b[4]};
                        default: mask <= {b[7], b[6], b[5], b[4]};
                    endcase
                    else h_we <= 1;
                    ps <= P_CMD;
                end else begin wt <= 0; ps <= P_RD; end
            end
            P_RD: begin wt <= wt + 1; if (wt == 2) begin txw <= rd[h_eng]; txn <= 0; ps <= P_TX; end end
            P_RUN: begin
                cycles <= cycles + 1; wt <= (wt == 3) ? 3 : wt + 1;          // let busy rise first
                if (wt == 3 && busy == 0) begin txw <= cycles; txn <= 0; ps <= P_TX; end
            end
            P_TX: begin
                pbyte <= txw[7:0]; push <= 1; txw <= txw >> 8; txn <= txn + 1;
                if (txn == 3) ps <= P_CMD;
            end
            default: ps <= P_CMD;
        endcase
    end
    assign led = {busy != 0, ps[2:0], rbusy};
endmodule
`default_nettype wire
