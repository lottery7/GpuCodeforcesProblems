// Author's solution of vector-addition on rtl: one element per cycle, as in judge/problems/examples/vector_add_rtl.
module solve (
    input  logic clk,
    input  logic start,
    output logic done,
    output logic [31:0] A_addr,
    input  logic [31:0] A_data,
    output logic [31:0] B_addr,
    input  logic [31:0] B_data,
    output logic [31:0] C_addr,
    output logic [31:0] C_data,
    output logic C_write,
    input  logic signed [31:0] N
);
    // IEEE 754 binary32 addition, rounding to nearest even; subnormals are flushed to zero.
    function automatic logic [31:0] fadd(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] x, y;
        logic [7:0] ex, ey, d;
        logic [27:0] X, Y, S;
        logic sticky, g, rs;
        logic [24:0] r;
        integer e, lz, i;
        if (a[30:0] >= b[30:0]) begin x = a; y = b; end else begin x = b; y = a; end
        ex = x[30:23]; ey = y[30:23];
        X = {1'b0, 1'b1, x[22:0], 3'b0};
        Y = ey == 0 ? 28'b0 : {1'b0, 1'b1, y[22:0], 3'b0};
        d = ex - ey;
        sticky = 0;
        for (i = 0; i < 27; i = i + 1) if (i < d && Y[i]) sticky = 1;
        Y = d >= 27 ? 28'b0 : Y >> d;
        Y[0] = Y[0] | sticky;
        e = ex;
        if (x[31] == y[31]) begin
            S = X + Y;
            if (S[27]) begin S = {1'b0, S[27:2], S[1] | S[0]}; e = e + 1; end
        end else begin
            S = X - Y;
            lz = 0;
            for (i = 0; i <= 26; i = i + 1) if (S[i]) lz = 26 - i;
            S = S << lz;
            e = e - lz;
        end
        g = S[2]; rs = S[1] | S[0];
        r = {1'b0, S[26:3]} + (g && (rs || S[3]));
        if (r[24]) begin r = r >> 1; e = e + 1; end
        if (ex == 0) fadd = {x[31] & y[31], 31'b0};
        else if (S == 0) fadd = 32'b0;
        else if (e <= 0) fadd = {x[31], 31'b0};
        else if (e >= 255) fadd = {x[31], 8'hFF, 23'b0};
        else fadd = {x[31], e[7:0], r[22:0]};
    endfunction

    logic [31:0] i, pending_index;
    logic running, pending, started;

    assign A_addr = i;
    assign B_addr = i;
    assign C_addr = pending_index;
    assign C_data = fadd(A_data, B_data);
    assign C_write = pending;

    always_ff @(posedge clk) begin
        if (start) begin
            i <= 0;
            running <= 1;
            started <= 1;
            pending <= 0;
            done <= 0;
        end else begin
            pending <= running && $signed(i) < N;
            pending_index <= i;
            if (running) begin
                if ($signed(i) < N) i <= i + 1;
                else running <= 0;
            end
            done <= started && !running && !pending;
        end
    end
endmodule
