// Author's solution of prefix-sum: the running sum is a register, and each element of C leaves on the cycle its
// element of A arrives.
module solve (
    input  logic clk,
    input  logic start,
    output logic done,
    input  logic [31:0] A_data,
    input  logic A_valid,
    output logic [31:0] C_data,
    output logic C_valid,
    input  logic signed [31:0] N
);
    logic [31:0] acc, count;

    assign C_data = acc + A_data;
    assign C_valid = A_valid;

    always_ff @(posedge clk) begin
        if (start) begin
            acc <= 0;
            count <= 0;
            done <= 0;
        end else if (A_valid) begin
            acc <= acc + A_data;
            count <= count + 1;
            if ($signed(count) == N - 1) done <= 1;
        end
    end
endmodule
