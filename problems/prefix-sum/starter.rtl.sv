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

    // A_data is the next element of A while A_valid is high, one element per cycle.
    // The judge takes C_data as the next element of C on every rising edge C_valid is high.
    assign C_data = 0;
    assign C_valid = 1'b0;
    assign done = 1'b0;

endmodule
