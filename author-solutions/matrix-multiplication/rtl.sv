// Author's solution of matrix-multiplication on rtl: one multiply-add per cycle. Each cycle asks for A[i][k] and
// B[k][j]; on the next one their product is added to the sum of C[i][j], which is written after k = K - 1.
// Addresses are kept as running sums, so the circuit has no integer multipliers.
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
    input  logic signed [31:0] M,
    input  logic signed [31:0] K,
    input  logic signed [31:0] N
);
    // IEEE 754 binary32, rounding to nearest even; subnormals are flushed to zero.
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

    function automatic logic [31:0] fmul(input logic [31:0] a, input logic [31:0] b);
        logic [47:0] p;
        logic [23:0] m;
        logic [24:0] r;
        logic g, rs, s;
        integer e;
        s = a[31] ^ b[31];
        p = {1'b1, a[22:0]} * {1'b1, b[22:0]};
        e = a[30:23] + b[30:23] - 127;
        if (p[47]) begin m = p[47:24]; g = p[23]; rs = |p[22:0]; e = e + 1; end
        else begin m = p[46:23]; g = p[22]; rs = |p[21:0]; end
        r = {1'b0, m} + (g && (rs || m[0]));
        if (r[24]) begin r = r >> 1; e = e + 1; end
        if (a[30:23] == 0 || b[30:23] == 0 || e <= 0) fmul = {s, 31'b0};
        else if (e >= 255) fmul = {s, 8'hFF, 23'b0};
        else fmul = {s, e[7:0], r[22:0]};
    endfunction

    // issue stage: indices of the next product
    logic [31:0] i, j, k, a_row, a_ptr, b_ptr, c_index;
    logic issuing, started;
    // product stage: what the data on A_data and B_data belongs to
    logic p_valid, p_first, p_last;
    logic [31:0] p_index, acc;
    logic [31:0] sum;

    assign A_addr = a_ptr;
    assign B_addr = b_ptr;
    assign sum = p_first ? fmul(A_data, B_data) : fadd(acc, fmul(A_data, B_data));
    assign C_addr = p_index;
    assign C_data = sum;
    assign C_write = p_valid && p_last;

    always_ff @(posedge clk) begin
        if (start) begin
            {i, j, k, a_row, a_ptr, b_ptr, c_index} <= '0;
            issuing <= 1;
            started <= 1;
            p_valid <= 0;
            done <= 0;
        end else begin
            p_valid <= issuing;
            p_first <= k == 0;
            p_last <= $signed(k) == K - 1;
            p_index <= c_index;
            if (p_valid) acc <= sum;
            if (issuing) begin
                if ($signed(k) == K - 1) begin
                    k <= 0;
                    c_index <= c_index + 1;
                    if ($signed(j) == N - 1) begin
                        j <= 0;
                        b_ptr <= 0;
                        a_row <= a_row + K;
                        a_ptr <= a_row + K;
                        if ($signed(i) == M - 1) issuing <= 0;
                        i <= i + 1;
                    end else begin
                        j <= j + 1;
                        b_ptr <= j + 1;
                        a_ptr <= a_row;
                    end
                end else begin
                    k <= k + 1;
                    a_ptr <= a_ptr + 1;
                    b_ptr <= b_ptr + N;
                end
            end
            done <= started && !issuing && !p_valid;
        end
    end
endmodule
